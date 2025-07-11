#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
这段代码实现了一个基于LSTM和Transformer的序列到序列模型（Seq2Seq），主要用于处理自然语言任务。以下是代码的功能分解：
初始化 (__init__)：
定义了LSTM层用于编码输入序列，双向LSTM将输入映射到隐藏状态。
使用Transformer的编码器和解码器进行特征提取和重构。
定义了一个线性层将隐藏状态映射回输入维度。
前向传播 (forward)：
输入序列通过LSTM层生成嵌入表示。
嵌入表示通过Transformer的编码器生成上下文表示。
如果启用重构模式 (recons=True)，则通过解码器生成输出，并计算重构损失。
重构损失 (recon_loss)：
计算输出与目标之间的负对数似然损失，忽略填充标记（PAD）。
'''

import math
import torch

from torch import nn
from torch.nn import functional as F

PAD = 0
UNK = 1
EOS = 2
SOS = 3
MASK = 4


class TrfmSeq2seq(nn.Module):
    def __init__(self, input_dim, hidden_size, num_head, n_layers, dropout, vocab_num, device, recons=False):
        super(TrfmSeq2seq, self).__init__()
        self.in_size = input_dim
        self.hidden_size = hidden_size
        self.embed = nn.LSTM(input_size=input_dim, hidden_size=int(hidden_size/2), bidirectional=True, batch_first=True,
                             num_layers=3, dropout=dropout)
        self.recons = recons
        self.device = device

        self.vocab_num = vocab_num
        transformer = nn.Transformer(d_model=hidden_size, nhead=num_head, num_encoder_layers=n_layers,
                                     num_decoder_layers=n_layers, dim_feedforward=hidden_size)
        self.encoder = transformer.encoder
        self.decoder = transformer.decoder
        self.out = nn.Linear(hidden_size, input_dim)

    def forward(self, src):
        # src: (T,B)\sout{}
        loss = 0
        embedded, _ = self.embed(src.to(self.device))  # (T,B,H)
        hidden = self.encoder(embedded)  # (T,B,H)
        if self.recons:
            out = self.decoder(embedded, hidden)
            out = self.out(out)  # (T,B,V)
            out = F.log_softmax(out, dim=-1)  # (T,B,V)
            loss = self.recon_loss(out, src.to(self.device), self.vocab_num)
        return loss, hidden  # (T,B,V)

    def recon_loss(self, output, target, vocab_max_num):
        loss = F.nll_loss(output.view(-1, vocab_max_num), target.contiguous().view(-1), ignore_index=PAD)
        return loss

