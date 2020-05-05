# encoding:utf-8
import time
import torch
import torch.nn as nn

from .text_cnn import TextCNN


class CaptionDiscriminator(nn.Module):
    def __init__(self, settings):
        super(CaptionDiscriminator, self).__init__()
        self.text_cnn = TextCNN(1, settings)
        self.weight_cliping()

    def forward(self, word_embs):
        # [bs, seq, word_dim]
        out = self.text_cnn(word_embs)  # [bs, 1]
        out = out.squeeze(1)  # [batch]
        return out

    def weight_cliping(self, limit=0.01):
        for p in self.parameters():
            p.data.clamp_(-limit, limit)

    def get_optim_criterion(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr), nn.BCEWithLogitsLoss()
