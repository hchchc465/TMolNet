
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from chemprop.features import BatchMolGraph
from chemprop.nn_utils import index_select_ND, get_activation_function
'''
这段代码实现了一个基于消息传递神经网络（MPNN）的分子图编码器 MPNEncoder，用于处理化学分子图数据。以下是主要功能分解：
初始化 (__init__)：
定义了原子和键的特征维度、隐藏层大小、深度、激活函数等超参数。
初始化线性变换层、激活函数、Dropout层以及GRU模块。
前向传播 (forward)：
输入为 BatchMolGraph 对象，提取原子和键的特征。
通过多层消息传递更新原子和键的隐藏状态。
使用 GRU 模块聚合信息，并返回最终的原子隐藏状态。
BatchGRU 类：
实现了批量化的双向 GRU 模块，用于处理不同长度的分子图数据。
包含填充（padding）、GRU 计算和去填充（unpadding）步骤。
'''
class MPNEncoder(nn.Module):
    def __init__(self, atom_fdim, bond_fdim, hidden_size, bias, depth, dropout, activation, device):
        super(MPNEncoder, self).__init__()
        self.atom_fdim = atom_fdim
        self.bond_fdim = bond_fdim
        self.hidden_size = hidden_size
        self.bias = bias
        self.depth = depth
        self.dropout = dropout
        self.device = device
        self.layers_per_message = 1

        # Dropout
        self.dropout_layer = nn.Dropout(p=self.dropout).to(device)

        # Activation
        self.act_func = get_activation_function(activation).to(device)

        # Input
        self.W_i_atom = nn.Linear(self.atom_fdim, self.hidden_size, bias=self.bias).to(device)
        self.W_i_bond = nn.Linear(self.bond_fdim, self.hidden_size, bias=self.bias).to(device)

        w_h_input_size_atom = self.hidden_size + self.bond_fdim
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias).to(device)

        w_h_input_size_bond = self.hidden_size

        for depth in range(self.depth - 1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias).to(device)

        self.W_o = nn.Linear((self.hidden_size) * 2, self.hidden_size).to(device)

        self.gru = BatchGRU(self.hidden_size).to(device)

        self.lr = nn.Linear(self.hidden_size * 3, self.hidden_size, bias=self.bias).to(device)

    def forward(self, mol_graph: BatchMolGraph, features_batch=None, batch_mask=None) -> torch.FloatTensor:

        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()

        f_atoms, f_bonds, a2b, b2a, b2revb = (
            f_atoms.to(self.device), f_bonds.to(self.device), a2b.to(self.device),
            b2a.to(self.device), b2revb.to(self.device)
        )

        def check_nan(name, x):
            """检查张量中是否含有 NaN、Inf，或是否全为 0，并打印具体位置信息"""
            if not isinstance(x, torch.Tensor):
                print(f"⚠️ {name} 不是 Tensor 类型，实际类型为 {type(x)}")
                return

            has_nan = torch.isnan(x)
            has_inf = torch.isinf(x)

            if has_nan.any():
                print(f"❌ GNN_model 里的 {name} 含 NaN，共 {has_nan.sum().item()} 个")
                nan_idx = torch.nonzero(has_nan)
                print(f"NaN 位置索引示例（最多显示前 5 个）:\n{nan_idx[:5]}")

            if has_inf.any():
                print(f"❌ GNN_model 里的 {name} 含 Inf，共 {has_inf.sum().item()} 个")
                inf_idx = torch.nonzero(has_inf)
                print(f"Inf 位置索引示例（最多显示前 5 个）:\n{inf_idx[:5]}")

            if torch.all(x == 0):
                print(f"⚠️ {name} 全为 0，形状为 {x.shape}")
            print("GNN_model已经检查")

        input_atom = self.act_func(self.W_i_atom(f_atoms))
        input_bond = self.act_func(self.W_i_bond(f_bonds))
        message_atom = input_atom.clone()
        message_bond = input_bond.clone()

        #check_nan("input_atom", input_atom)
        #check_nan("input_bond", input_bond)

        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            #check_nan(f"agg_message@{depth}", agg_message)

            agg_sum = agg_message.sum(dim=1)
            agg_max = agg_message.max(dim=1)[0]
            #check_nan(f"agg_sum@{depth}", agg_sum)
            #check_nan(f"agg_max@{depth}", agg_max)

            message_atom = message_atom + agg_sum * agg_max
            #check_nan(f"message_atom@{depth}", message_atom)

            rev_message = message_bond[b2revb]
            message_bond = message_atom[b2a] - rev_message
            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout_layer(self.act_func(input_bond + message_bond))

            #check_nan(f"message_bond@{depth}", message_bond)

        agg_message = index_select_ND(message_bond, a2b)
        agg_sum = agg_message.sum(dim=1)
        agg_max = agg_message.max(dim=1)[0]
        message_concat = torch.cat([agg_sum * agg_max, message_atom, input_atom], 1)
        agg_message = self.lr(message_concat)
        agg_message = self.gru(agg_message, a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))
        atom_hiddens = self.dropout_layer(atom_hiddens)

        #check_nan("atom_hiddens", atom_hiddens)

        return atom_hiddens[1:]

class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        return message

