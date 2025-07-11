#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn as nn

from utils.compound_tools import CompoundKit
from models_lib.basic_block import RBF


class AtomEmbedding(torch.nn.Module):
    """
    Atom Encoder
    # 代码解释
这段代码定义了多个用于分子图表示学习的嵌入模块，主要包括原子、键和键角的特征编码。以下是各部分的功能解释：

1. **AtomEmbedding**：对离散的原子特征进行嵌入编码，通过`nn.Embedding`将每个原子特征映射到固定维度的向量空间。
2. **AtomFloatEmbedding**：对连续的原子浮点特征进行编码，使用径向基函数（RBF）将特征转换为高维空间，并通过线性层映射到嵌入空间。
3. **BondEmbedding**：对离散的键特征进行嵌入编码，类似于`AtomEmbedding`。
4. **BondFloatRBF**：对连续的键浮点特征进行编码，类似于`AtomFloatEmbedding`。
5. **BondAngleFloatRBF**：对键角的连续浮点特征进行编码，同样使用RBF和线性层。
    """

    def __init__(self, atom_names, embed_dim, device):
        super(AtomEmbedding, self).__init__()
        self.atom_names = atom_names

        self.embed_list = nn.ModuleList()
        for name in self.atom_names:
            embed = nn.Embedding(CompoundKit.get_atom_feature_size(name) + 5, embed_dim).to(device)
            self.embed_list.append(embed)

    def forward(self, node_features):
        """
        Args:
            node_features(dict of tensor): node features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_names):
            out_embed += self.embed_list[i](node_features[i])
        return out_embed


class AtomFloatEmbedding(torch.nn.Module):
    """
    Atom Float Encoder
    """

    def __init__(self, atom_float_names, embed_dim, rbf_params=None, device=None):
        super(AtomFloatEmbedding, self).__init__()
        self.atom_float_names = atom_float_names

        if rbf_params is None:
            self.rbf_params = {
                'van_der_waals_radis': (torch.arange(1, 3, 0.2), 10.0),  # (centers, gamma)
                'partial_charge': (torch.arange(-1, 4, 0.25), 10.0),  # (centers, gamma)
                'mass': (torch.arange(0, 2, 0.1), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params
            self.linear_list = nn.ModuleList()
            self.rbf_list = nn.ModuleList()
            for name in self.atom_float_names:
                centers, gamma = self.rbf_params[name]
                rbf = RBF(centers, gamma).to(device)
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim).to(device)
                self.linear_list.append(linear)
                self.rbf_list.append(rbf)
                linear = nn.Linear(len(centers), embed_dim).to(device)
                self.linear_list.append(linear)

    def forward(self, feats):
        """
        Args:
            feats(dict of tensor): node float features.
        """
        out_embed = 0
        for i, name in enumerate(self.atom_float_names):
            x = feats[name]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x)
        return out_embed


class BondEmbedding(nn.Module):
    """
    Bond Encoder
    """

    def __init__(self, bond_names, embed_dim, device):
        super(BondEmbedding, self).__init__()
        self.bond_names = bond_names

        self.embed_list = nn.ModuleList()
        for name in self.bond_names:
            embed = nn.Embedding(CompoundKit.get_bond_feature_size(name) + 5, embed_dim).to(device)
            self.embed_list.append(embed)

    def forward(self, edge_features):
        """
        Args:
            edge_features(dict of tensor): edge features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_names):
            out_embed += self.embed_list[i](edge_features[i].long())
        return out_embed


class BondFloatRBF(nn.Module):
    """
    Bond Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_float_names, embed_dim, rbf_params=None, device=None):
        super(BondFloatRBF, self).__init__()
        self.bond_float_names = bond_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_length': (torch.arange(0, 2, 0.1).to(device), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma).to(device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)

    def forward(self, bond_float_features):
        """
        Args:
            bond_float_features(dict of tensor): bond float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_float_names):
            x = bond_float_features[i]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x.float())
        return out_embed


class BondAngleFloatRBF(nn.Module):
    """
    Bond Angle Float Encoder using Radial Basis Functions
    """

    def __init__(self, bond_angle_float_names, embed_dim, rbf_params=None, device=None):
        super(BondAngleFloatRBF, self).__init__()
        self.bond_angle_float_names = bond_angle_float_names

        if rbf_params is None:
            self.rbf_params = {
                'bond_angle': (torch.arange(0, np.pi, 0.1).to(device), 10.0),  # (centers, gamma)
            }
        else:
            self.rbf_params = rbf_params

        self.linear_list = nn.ModuleList()
        self.rbf_list = nn.ModuleList()
        for name in self.bond_angle_float_names:
            centers, gamma = self.rbf_params[name]
            rbf = RBF(centers, gamma).to(device)
            self.rbf_list.append(rbf)
            linear = nn.Linear(len(centers), embed_dim).to(device)
            self.linear_list.append(linear)

    def forward(self, bond_angle_float_features):
        """
        Args:a
            bond_angle_float_features(dict of tensor): bond angle float features.
        """
        out_embed = 0
        for i, name in enumerate(self.bond_angle_float_names):
            x = bond_angle_float_features[i]
            rbf_x = self.rbf_list[i](x)
            out_embed += self.linear_list[i](rbf_x.float())
        return out_embed