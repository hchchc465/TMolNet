
# coding: utf-8
import os
import copy
import json
import logging
import warnings
import numpy as np
import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader, RandomSampler, BatchSampler
from tensorboardX import SummaryWriter
from parser_args import get_args
from chemprop.data import StandardScaler
from chemprop.nn_utils import NoamLR
from chemprop.features import mol2graph, get_atom_fdim, get_bond_fdim
from chemprop.data.utils import get_class_sizes
from utils.dataset import Seq2seqDataset, get_data, split_data, MoleculeDataset, InMemoryDataset
from utils.evaluate import eval_rocauc, eval_rmse
from build_vocab import WordVocab
from models_lib.multi_modal import Multi_modal
from featurizers.gem_featurizer import GeoPredTransformFn
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.Chem.rdMolDescriptors import GetHashedAtomPairFingerprintAsBitVect

PAD, UNK, EOS, SOS, MASK = 0, 1, 2, 3, 4
warnings.filterwarnings('ignore')

def load_json_config(path):
    return json.load(open(path, 'r'))

def check_stall(validation_result, current_epoch, task_type='reg'):
    if not hasattr(check_stall, 'history'):
        check_stall.history, check_stall.best_epoch = [], 0
        check_stall.best_result = float('inf') if task_type == 'reg' else -float('inf')
    check_stall.history.append(validation_result)
    if (task_type == 'reg' and validation_result < check_stall.best_result) or        (task_type != 'reg' and validation_result > check_stall.best_result):
        check_stall.best_result = validation_result
        check_stall.best_epoch = current_epoch
    return current_epoch - check_stall.best_epoch >= 10

def calculate_combined_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        maccs = np.array(list(MACCSkeys.GenMACCSKeys(mol).ToBitString()), dtype=int) if mol else np.zeros(167)
        pubchem = np.array(list(GetHashedAtomPairFingerprintAsBitVect(mol, nBits=881).ToBitString()), dtype=int) if mol else np.zeros(881)
        pharmacophore = np.zeros(200, dtype=int)
        if mol:
            erg_fp = AllChem.GetErGFingerprint(mol)
            for i in range(min(len(erg_fp), 200)):
                pharmacophore[i] = erg_fp[i]
        return np.concatenate([maccs, pubchem, pharmacophore])
    except Exception:
        return np.zeros(1248, dtype=int)

def prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, fingerprints, device):
    fp_batch = fingerprints[idx].to(device)
    edge_batch1, edge_batch2 = [], []
    geo_gen = geo_data.get_batch(idx)
    node_id_all = [geo_gen[0].batch, geo_gen[1].batch]
    for i in range(geo_gen[0].num_graphs):
        edge_batch1.append(torch.ones(geo_gen[0][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
        edge_batch2.append(torch.ones(geo_gen[1][i].edge_index.shape[1], dtype=torch.long).to(device) * i)
    edge_id_all = [torch.cat(edge_batch1), torch.cat(edge_batch2)]

    mol_batch = MoleculeDataset([gnn_data[i] for i in idx])
    smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
    gnn_batch = mol2graph(smiles_batch, args)
    batch_mask_seq, batch_mask_gnn = [], []
    for i, (smile, mol) in enumerate(zip(smiles_batch, mol_batch.mols())):
        batch_mask_seq.append(torch.ones(len(smile), dtype=torch.long).to(device) * i)
        batch_mask_gnn.append(torch.ones(mol.GetNumAtoms(), dtype=torch.long).to(device) * i)
    batch_mask_seq = torch.cat(batch_mask_seq)
    batch_mask_gnn = torch.cat(batch_mask_gnn)
    mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch]).to(device)
    targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch]).to(device)
    return seq_data[idx], seq_mask[idx], batch_mask_seq, gnn_batch, features_batch, batch_mask_gnn, geo_gen, node_id_all, edge_id_all, mask, targets, fp_batch

def train(args, model, optimizer, scheduler, train_loader, seq_data, seq_mask, gnn_data, geo_data, fingerprints, device):
    total_all_loss, total_lab_loss, total_cl_loss = 0, 0, 0
    model.train()
    for idx in tqdm(train_loader):
        model.zero_grad()
        data = prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, fingerprints, device)
        x_list, preds, gate_weights = model(*data[:-2], data[-1])
        all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds, data[-2], data[-3], gate_weights, args.alpha, args.beta, args.T)
        total_all_loss += all_loss.item()
        total_lab_loss += lab_loss.item()
        total_cl_loss += cl_loss.item()
        all_loss.backward()
        optimizer.step()
        if isinstance(scheduler, NoamLR):
            scheduler.step()
    return total_all_loss, total_lab_loss, total_cl_loss

@torch.no_grad()
def evaluate(args, model, scaler, data_loader, seq_data, seq_mask, gnn_data, geo_data, fingerprints, device):
    total_all_loss, total_lab_loss, total_cl_loss = 0, 0, 0
    y_true, y_pred = [], []
    model.eval()
    for idx in data_loader:
        data = prepare_data(args, idx, seq_data, seq_mask, gnn_data, geo_data, fingerprints, device)
        x_list, preds, gate_weights = model(*data[:-2], data[-1])
        if scaler and args.task_type == 'reg':
            preds = torch.tensor(scaler.inverse_transform(preds.detach().cpu().numpy())).to(device)
        all_loss, lab_loss, cl_loss = model.loss_cal(x_list, preds, data[-2], data[-3], gate_weights, args.alpha, args.beta, args.T)
        total_all_loss += all_loss.item()
        total_lab_loss += lab_loss.item()
        total_cl_loss += cl_loss.item()
        y_true.append(data[-2])
        y_pred.append(preds)
    y_true = torch.cat(y_true).cpu().numpy()
    y_pred = torch.cat(y_pred).cpu().numpy()
    metric = eval_rocauc if args.task_type == 'class' else eval_rmse
    result = metric({"y_true": y_true, "y_pred": y_pred})
    return result, total_all_loss, total_lab_loss, total_cl_loss

def init_logger(log_dir, args):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(log_dir, f"Train_{args.seed}"))
    log_file = os.path.join(log_dir, f"{args.dataset}_{args.lr}_{args.cl_loss}_{args.epochs}_{args.batch_size}_Train_{args.seed}.log")
    logging.basicConfig(level=logging.INFO, filename=log_file, filemode='w',
                        format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(log_file))
    return writer, logger

def get_device(args, logger):
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.cuda else 'cpu')
    logger.info(f"Using device: {device}")
    return device

def main(args):
    logs_dir = f'./LOG/{args.dataset}_{args.lr}_{args.cl_loss}_{args.epochs}_{args.batch_size}/'
    writer, logger = init_logger(logs_dir, args)
    device = get_device(args, logger)

    datas, args.seq_len = get_data(f'data/{args.dataset}.csv', args)
    args.output_dim = datas.num_tasks()
    args.gnn_atom_dim = get_atom_fdim(args)
    args.gnn_bond_dim = get_bond_fdim(args) + (not args.atom_messages) * args.gnn_atom_dim
    args.features_size = datas.features_size()

    train_data, val_data, test_data = split_data(data=datas, split_type=args.split_type, sizes=args.split_sizes, seed=args.seed, args=args)
    train_idx, val_idx, test_idx = [d.idx for d in train_data], [d.idx for d in val_data], [d.idx for d in test_data]

    fingerprints = torch.tensor(np.array([calculate_combined_fingerprint(s) for s in tqdm(datas.smiles())]),
                                dtype=torch.float32).to(device)
    vocab = WordVocab.load_vocab(f'./data/{args.dataset}_vocab.pkl')
    args.seq_input_dim = args.vocab_num = len(vocab)
    seq = Seq2seqDataset(datas.smiles(), vocab, args.seq_len, device)
    seq_data = torch.stack([x[1] for x in seq])
    seq_mask = torch.zeros(len(datas), args.seq_len).bool().to(device)
    for i, smile in enumerate(datas.smiles()):
        seq_mask[i, 1:1 + len(smile)] = True

    compound_encoder_config = load_json_config(args.compound_encoder_config)
    model_config = load_json_config(args.model_config)
    if args.dropout_rate is not None:
        compound_encoder_config['dropout_rate'] = args.dropout_rate

    transform_fn = GeoPredTransformFn(model_config['pretrain_tasks'], model_config['mask_ratio'])
    data_3d = InMemoryDataset(datas.smiles())
    if not os.path.exists(f'./data/{args.dataset}/'):
        data_3d.transform(transform_fn, num_workers=1)
        data_3d.save_data(f'./data/{args.dataset}/')
    else:
        data_3d = data_3d._load_npz_data_path(f'./data/{args.dataset}/')
        data_3d = InMemoryDataset(data_3d)
    data_3d.get_data(device)

    train_loader = DataLoader(train_idx, batch_size=args.batch_size, sampler=RandomSampler(train_idx), drop_last=True)
    val_loader = BatchSampler(val_idx, batch_size=args.batch_size, drop_last=False)
    test_loader = BatchSampler(test_idx, batch_size=args.batch_size, drop_last=False)

    scaler = None
    if args.task_type == 'reg':
        scaler = StandardScaler().fit(train_data.targets())
        scaled_targets = scaler.transform(train_data.targets())
        for idx, target in zip(train_idx, scaled_targets):
            datas[idx].set_targets(target)

    args.seq_hidden_dim = args.gnn_hidden_dim
    args.geo_hidden_dim = args.gnn_hidden_dim
    model = Multi_modal(args, compound_encoder_config, device)
    optimizer = Adam(model.parameters(), lr=args.init_lr, weight_decay=1e-5)
    scheduler = NoamLR(optimizer, [args.warmup_epochs], [args.epochs], len(train_loader), [args.init_lr], [args.max_lr], [args.final_lr])

    best_result, best_epoch = None, 0
    for epoch in range(args.epochs):
        total_all_loss, total_lab_loss, total_cl_loss = train(args, model, optimizer, scheduler, train_loader, seq_data, seq_mask, datas, data_3d, fingerprints, device)
        val_result, val_all_loss, val_lab_loss, val_cl_loss = evaluate(args, model, scaler, val_loader, seq_data, seq_mask, datas, data_3d, fingerprints, device)
        val_score = val_result['rocauc'] if args.task_type == 'class' else val_result['rmse']

        writer.add_scalars('loss', {
            'train_all_loss': total_all_loss,
            'train_lab_loss': total_lab_loss,
            'train_cl_loss': total_cl_loss,
            'val_all_loss': val_all_loss,
            'val_lab_loss': val_lab_loss,
            'val_cl_loss': val_cl_loss
        }, epoch + 1)
        writer.add_scalar('val_result', val_score, epoch + 1)

        logger.info(f"Epoch {epoch}, Val Score: {val_score:.4f}")
        if check_stall(val_score, epoch, args.task_type):
            logger.info(f"Early stopping at epoch {epoch}")
            break
    return best_result

if __name__ == '__main__':
    arg = get_args()
    main(arg)
