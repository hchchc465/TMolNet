import argparse

def get_args():
    parser = argparse.ArgumentParser(description='SGG PyTorch Implementation')

    # === Graph Settings ===
    parser.add_argument('--gnn_atom_dim', type=int, default=0)
    parser.add_argument('--gnn_bond_dim', type=int, default=0)
    parser.add_argument('--gnn_activation', type=str, default='ReLU')
    parser.add_argument('--gnn_num_layers', type=int, default=5)
    parser.add_argument('--gnn_hidden_dim', type=int, default=256)
    parser.add_argument('--atom_messages', action='store_true', default=False)

    # === Sequence Settings ===
    parser.add_argument('--seq_len', type=int, default=220)
    parser.add_argument('--seq_input_dim', type=int, default=64)
    parser.add_argument('--seq_num_heads', type=int, default=4)
    parser.add_argument('--seq_num_layers', type=int, default=4)
    parser.add_argument('--seq_hidden_dim', type=int, default=256)

    # === Geometry Settings ===
    parser.add_argument('--geometry', type=bool, default=True)
    parser.add_argument('--geo_hidden_dim', type=int, default=256)
    parser.add_argument('--dropout_rate', type=float, default=0.5)
    parser.add_argument('--model_config', type=str, default="./model_configs/pretrain_gem.json")
    parser.add_argument('--compound_encoder_config', type=str, default="./model_configs/geognn_l8.json")

    # === Fusion and Fingerprint ===
    parser.add_argument('--fingerprint', type=bool, default=True)
    parser.add_argument('--fusion', type=int, default=3)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=1)

    # === Training Settings ===
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--init_lr', type=float, default=1e-3)
    parser.add_argument('--max_lr', type=float, default=2e-3)
    parser.add_argument('--final_lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--warmup_epochs', type=float, default=2.0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--cl_loss', type=float, default=0.1)
    parser.add_argument('--cl_loss_num', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.08)
    parser.add_argument('--beta', type=float, default=0.08)
    parser.add_argument('--T', type=float, default=0.1)
    parser.add_argument('--pro_num', type=int, default=1)
    parser.add_argument('--pool_type', type=str, default='attention')

    # === General Options ===
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='esol')
    parser.add_argument('--task_type', type=str, default='reg', choices=['reg', 'class'])
    parser.add_argument('--split_type', type=str, default='scaffold_balanced',
                        choices=['random', 'scaffold_balanced', 'predetermined', 'crossval', 'index_predetermined'])
    parser.add_argument('--split_sizes', type=float, nargs=3, default=[0.8, 0.1, 0.1])
    parser.add_argument('--num_tasks', type=int, default=1)
    parser.add_argument('--vocab_num', type=int, default=0)

    args = parser.parse_args()
    return args
