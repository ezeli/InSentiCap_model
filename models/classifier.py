# encoding:utf-8
import torch
import torch.nn as nn

from .text_cnn import TextCNN


class SentimentClassifier(nn.Module):
    def __init__(self, num_senti, settings):
        super(SentimentClassifier, self).__init__()
        self.text_cnn = TextCNN(num_senti, settings)

    def forward(self, word_embs):
        # [bs, seq, word_dim]
        out = self.text_cnn(word_embs)  # [batch, num_senti]
        return out

    def weight_cliping(self, limit=0.01):
        for p in self.parameters():
            p.data.clamp_(-limit, limit)

    def get_optim_criterion(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr), nn.CrossEntropyLoss()
