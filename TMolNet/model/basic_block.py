#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

import torch.nn as nn


class Activation(nn.Module):
    """
    Activation
    # 代码解释
这段代码定义了三个神经网络相关的类：`Activation`、`RBF` 和 `MLP`，分别用于实现激活函数、径向基函数（RBF）和多层感知机（MLP）。具体功能如下：

1. **Activation**：根据输入的激活函数类型（如ReLU或LeakyReLU），动态生成对应的激活层。
2. **RBF**：实现径向基函数，通过计算输入与中心点的距离并应用指数衰减公式，输出特征映射。
3. **MLP**：构建一个多层感知机，支持指定层数、隐藏层大小、激活函数类型和dropout率。

    """

    def __init__(self, act_type, **params):
        super(Activation, self).__init__()
        if act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'leaky_relu':
            self.act = nn.LeakyReLU(**params)
        else:
            raise ValueError(act_type)

    def forward(self, x):
        """tbd"""
        return self.act(x)


class RBF(nn.Module):
    """
    Radial Basis Function
    """
    def __init__(self, centers, gamma, dtype='float32'):
        super(RBF, self).__init__()
        self.centers = torch.reshape(torch.tensor(centers), [1, -1])
        self.gamma = gamma

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, 1).
        Returns:
            y(tensor): (-1, n_centers)
        """
        x = torch.reshape(x, [-1, 1])
        return torch.exp(-self.gamma * torch.square(x - self.centers))

class MLP(nn.Module):
    """
    MLP
    """
    def __init__(self, layer_num, in_size, hidden_size, out_size, act, dropout_rate):
        super(MLP, self).__init__()

        layers = []
        for layer_id in range(layer_num):
            if layer_id == 0:
                layers.append(nn.Linear(in_size, hidden_size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(Activation(act))
            elif layer_id < layer_num - 1:
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.Dropout(dropout_rate))
                layers.append(Activation(act))
            else:
                layers.append(nn.Linear(hidden_size, out_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x(tensor): (-1, dim).
        """
        return self.mlp(x)