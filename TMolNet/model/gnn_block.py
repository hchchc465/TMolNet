#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import warnings

import torch.nn as nn

from torch_geometric.nn.glob import global_add_pool, global_mean_pool, global_max_pool


def segment_pool(data, segment_ids, pool_type, name=None):
    """
    Segment Operator.
    这段代码实现了一个基于几何图神经网络（GeoGNN）的模型，主要用于处理分子图数据。以下是代码的功能分解：
GeoGNNBlock 类：定义了一个图神经网络的基本块，包含 GINEConv 层、归一化层和 dropout 层，用于更新节点特征。
GeoGNNModel 类：
初始化部分：配置嵌入维度、dropout 率、层数等超参数，并初始化原子和键的嵌入层。
前向传播部分：通过多层 GeoGNNBlock 更新节点和边的特征，最终输出节点和边的表示。
    """
    pool_type = pool_type.upper()
    if pool_type == "SUM":
        return global_add_pool(data, segment_ids)
    elif pool_type == "MEAN":
        return global_mean_pool(data, segment_ids)
    elif pool_type == "MAX":
        return global_max_pool(data, segment_ids)
    else:
        raise ValueError(
            "We only support sum, mean, max, min pool types in segment_pool function."
        )


class GraphPool(nn.Module):
    """Implementation of graph pooling

    This is an implementation of graph pooling

    Args:
        pool_type: The type of pooling ("sum", "mean" , "min", "max"). Default:None

    """

    def __init__(self, pool_type=None):
        super(GraphPool, self).__init__()
        self.pool_type = pool_type

    def forward(self, graph, feature, node_id, edge_id, pool_type=None):
        """
         Args:
            graph: the graph object from (:code:`Graph`)

            feature: A tensor with shape (num_nodes, feature_size).

            pool_type: The type of pooling ("sum", "mean" , "min", "max"). Default:None
        Return:
            A tensor with shape (num_graph, feature_size)
        """
        if pool_type is not None:
            warnings.warn("The pool_type (%s) argument in forward function " \
                    "will be discarded in the future, " \
                    "please initialize it when creating a GraphPool instance.")
        else:
            pool_type = self.pool_type
        graph_feat = segment_pool(feature, node_id, pool_type)
        return graph_feat


class GraphNorm(nn.Module):
    """Implementation of graph normalization. Each node features is divied by sqrt(num_nodes) per graphs.

    Args:
        graph: the graph object from (:code:`Graph`)
        feature: A tensor with shape (num_nodes, feature_size).

    Return:
        A tensor with shape (num_nodes, hidden_size)

    References:

    [1] BENCHMARKING GRAPH NEURAL NETWORKS. https://arxiv.org/abs/2003.00982

    """

    def __init__(self, device):
        super(GraphNorm, self).__init__()
        self.device = device
        self.graph_pool = GraphPool(pool_type="sum")

    def forward(self, graph, feature, node_id, edge_id):
        """graph norm"""
        nodes = torch.ones(size=[graph.num_nodes, 1]).to(self.device)
        norm = self.graph_pool(graph, nodes, node_id, edge_id)
        norm = torch.sqrt(norm).to(self.device)
        norm = torch.gather(norm, dim=0, index=node_id.unsqueeze(dim=1))
        return feature / norm


class MeanPool(nn.Module):
    """
    TODO: temporary class due to pgl mean pooling
    """
    def __init__(self):
        super().__init__()
        self.graph_pool = GraphPool(pool_type="sum")

    def forward(self, graph, node_feat, node_id, edge_id):
        """
        mean pooling
        """
        sum_pooled = self.graph_pool(graph, node_feat, node_id, edge_id)
        ones_sum_pooled = self.graph_pool(
            graph, torch.ones_like(node_feat).to(self.device), node_id, edge_id)
        pooled = sum_pooled / ones_sum_pooled
        return pooled

