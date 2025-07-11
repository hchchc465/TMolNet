#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import init, Parameter
from torch_geometric.nn import global_mean_pool, GlobalAttention

from models_lib.gnn_model import MPNEncoder
from models_lib.gem_model import GeoGNNModel
from models_lib.seq_model import TrfmSeq2seq
from models_lib.fp_encode import FingerprintEncoder

# 根据任务类型选择损失函数
loss_type = {'class': nn.BCEWithLogitsLoss(reduction="none"),
             'reg': nn.MSELoss(reduction="none")}


class ModalityGating(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(4 * emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4)

    def forward(self, Hs, Hg, Hh, Hfp):
        cat = torch.cat([Hs, Hg, Hh, Hfp], dim=-1)
        x = F.relu(self.fc1(cat))
        weights = F.softmax(self.fc2(x), dim=-1)
        w_s, w_g, w_h, w_fp = weights.split(1, dim=-1)
        fused = w_s * Hs + w_g * Hg + w_h * Hh + w_fp * Hfp
        return fused, weights


class TaskAwareModalityGating(nn.Module):
    def __init__(self, emb_dim: int, hidden_dim: int = 512, task_dim: int = 8):
        super().__init__()
        input_dim = 4 * emb_dim + task_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 4)

    def forward(self, Hs, Hg, Hh, Hfp, task_embed=None):
        cat = torch.cat([Hs, Hg, Hh, Hfp], dim=-1)
        if task_embed is not None:
            if task_embed.dim() == 1:
                task_embed = task_embed.unsqueeze(0).expand(Hs.size(0), -1)
            cat = torch.cat([cat, task_embed], dim=-1)
        x = F.relu(self.fc1(cat))
        weights = F.softmax(self.fc2(x), dim=-1)
        w_s, w_g, w_h, w_fp = weights.split(1, dim=-1)
        fused = w_s * Hs + w_g * Hg + w_h * Hh + w_fp * Hfp
        return fused, weights


class TaskEmbedding(nn.Module):
    def __init__(self, num_tasks: int, task_dim: int, mode: str = 'add'):
        super().__init__()
        assert mode in ['add', 'concat']
        self.mode = mode
        self.task_num_embed = nn.Embedding(num_tasks + 1, task_dim)
        self.task_type_embed = nn.Embedding(3, task_dim)
        self.output_dim = task_dim if mode == 'add' else 2 * task_dim

    def forward(self, task_nums, task_types):
        task_num_emb = self.task_num_embed(task_nums)
        task_type_emb = self.task_type_embed(task_types)
        return task_num_emb + task_type_emb if self.mode == 'add' else torch.cat([task_num_emb, task_type_emb], dim=-1)


class Global_Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.at = GlobalAttention(gate_nn=torch.nn.Linear(hidden_size, 1))

    def forward(self, x, batch):
        return self.at(x, batch)


class WeightFusion(nn.Module):
    def __init__(self, feat_views, feat_dim, bias: bool = True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = Parameter(torch.empty((1, 1, feat_views), **factory_kwargs))
        self.bias = Parameter(torch.empty(int(feat_dim), **factory_kwargs)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return sum([input[i] * weight for i, weight in enumerate(self.weight[0][0])]) + (self.bias if self.bias is not None else 0)


class TMolNet(nn.Module):
    def __init__(self, args, compound_encoder_config, device):
        super().__init__()
        self.args = args
        self.device = device
        self.latent_dim = args.latent_dim
        self.batch_size = args.batch_size
        self.fingerprint = args.fingerprint
        self.graph = args.graph
        self.sequence = args.sequence
        self.geometry = args.geometry
        self.task_num = args.output_dim
        self.task_dim = args.task_dim
        self.Comparison = args.Comparison
        self.entropy = args.entropy

        self.gnn = MPNEncoder(atom_fdim=args.gnn_atom_dim, bond_fdim=args.gnn_bond_dim,
                              hidden_size=args.gnn_hidden_dim, bias=args.bias, depth=args.gnn_num_layers,
                              dropout=args.dropout, activation=args.gnn_activation, device=device)

        self.transformer = TrfmSeq2seq(input_dim=args.seq_input_dim, hidden_size=args.seq_hidden_dim,
                                       num_head=args.seq_num_heads, n_layers=args.seq_num_layers,
                                       dropout=args.dropout, vocab_num=args.vocab_num,
                                       device=device, recons=args.recons).to(device)

        self.compound_encoder = GeoGNNModel(args, compound_encoder_config, device)
        self.fp_encoder = FingerprintEncoder().to(device)

        self.pro_seq = self._build_proj(args.seq_hidden_dim)
        self.pro_gnn = self._build_proj(args.gnn_hidden_dim)
        self.pro_geo = self._build_proj(args.geo_hidden_dim)

        self.entropy = loss_type[args.task_type]

        self.pool = global_mean_pool if args.pool_type == 'mean' else Global_Attention(args.seq_hidden_dim).to(device)

        fusion_dim = args.gnn_hidden_dim * self.graph + args.seq_hidden_dim * self.sequence + args.geo_hidden_dim * self.geometry
        if args.fusion == 3:
            fusion_dim /= (self.graph + self.sequence + self.geometry)
            self.fusion = WeightFusion(self.graph + self.sequence + self.geometry, fusion_dim, device=device)
        elif args.fusion in [0, 2]:
            fusion_dim = args.seq_hidden_dim

        self.dropout = nn.Dropout(args.dropout)
        self.task_embedding = TaskEmbedding(self.task_num, self.task_dim).to(device)
        self.modality_gating = TaskAwareModalityGating(emb_dim=256, hidden_dim=512).to(device)

        self.output_layer = nn.Sequential(
            nn.Linear(int(fusion_dim), int(fusion_dim)), nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(int(fusion_dim), args.output_dim)
        ).to(device)

    def _build_proj(self, in_dim):
        return nn.Sequential(
            nn.Linear(in_dim, self.latent_dim), nn.ReLU(inplace=True),
            nn.Linear(self.latent_dim, self.latent_dim)
        ).to(self.device)

    def forward(self, trans_batch_seq, seq_mask, batch_mask_seq, gnn_batch_graph, gnn_feature_batch,
                batch_mask_gnn, graph_dict, node_id_all, edge_id_all, fp_batch):
        x_list, cl_list = [], []

        if self.graph:
            gnn_feat = self.gnn(gnn_batch_graph, gnn_feature_batch, batch_mask_gnn)
            graph_gnn_x = self.pool(gnn_feat, batch_mask_gnn)
            if self.args.norm: graph_gnn_x = F.normalize(graph_gnn_x, p=2, dim=1)
            x_list.append(graph_gnn_x)
            cl_list.append(self.pro_gnn(graph_gnn_x))

        if self.sequence:
            _, node_seq_x = self.transformer(trans_batch_seq)
            graph_seq_x = self.pool(node_seq_x[seq_mask], batch_mask_seq)
            if self.args.norm: graph_seq_x = F.normalize(graph_seq_x, p=2, dim=1)
            x_list.append(graph_seq_x)
            cl_list.append(self.pro_seq(graph_seq_x))

        if self.fingerprint:
            fp_feat = self.fp_encoder(fp_batch)
            x_list.append(fp_feat)
            cl_list.append(self.pro_gnn(fp_feat))

        if self.geometry:
            node_repr, _ = self.compound_encoder(graph_dict[0], graph_dict[1], node_id_all, edge_id_all)
            graph_geo_x = self.pool(node_repr, node_id_all[0])
            if self.args.norm: graph_geo_x = F.normalize(graph_geo_x, p=2, dim=1)
            x_list.append(graph_geo_x)
            cl_list.append(self.pro_geo(graph_geo_x.to(self.device)))

        task_type = 0 if self.args.task_type == "class" else 1
        task_type = torch.full((x_list[0].shape[0],), task_type, dtype=torch.long, device=self.device)
        task_num = torch.full((x_list[0].shape[0],), self.task_num - 1, dtype=torch.long, device=self.device)
        task_emb = self.task_embedding(task_num, task_type)

        molecule_emb, gate_weights = self.modality_gating(x_list[0], x_list[1], x_list[2], x_list[3], task_emb)
        pred = self.output_layer(molecule_emb)
        return cl_list, molecule_emb, gate_weights, pred

    def label_loss(self, pred, label, mask):
        return self.entropy(pred, label).sum() / mask.sum()

    def cl_loss(self, x1, x2, T=0.1):
        sim = torch.einsum('ik,jk->ij', x1, x2) / (
              torch.einsum('i,j->ij', x1.norm(dim=1), x2.norm(dim=1)) + 1e-8)
        sim = torch.exp(sim / T)
        pos_sim = sim[range(x1.size(0)), range(x1.size(0))]
        loss = -torch.log(pos_sim / (sim.sum(dim=1) - pos_sim + 1e-8)).mean()
        return loss

    def compute_modal_entropy(self, weights, eps=1e-8):
        return -(weights * torch.log(weights + eps)).sum(dim=1).mean()

    def loss_cal(self, x_list, pred, label, mask, gate_weights, alpha=0.08, beta=0.08, T=0.1):
        loss1 = self.label_loss(pred, label, mask)
        loss2 = sum([self.cl_loss(x_list[i], x_list[(i - 1) % len(x_list)], T)
                     for i in range(len(x_list))]) if self.Comparison else torch.tensor(0.).to(self.device)
        loss3 = self.compute_modal_entropy(gate_weights) if self.entropy else torch.tensor(0.).to(self.device)
        return loss1 + alpha * loss2 - beta * loss3, loss1, loss2
