# encoding:utf-8
import torch
import torch.nn as nn

from .text_cnn import TextCNN


class BackTranslator(nn.Module):
    def __init__(self, num_senti, settings):
        super(BackTranslator, self).__init__()
        self.senti_embed = nn.Sequential(
            nn.Embedding(num_senti, settings['word_emb_dim']),
            nn.ReLU())
        self.text_cnn = TextCNN(settings['fc_feat_dim'], settings)

    def forward(self, word_embs, senti_labels):
        # [bs, seq, word_dim], [bs]
        senti_embs = self.senti_embed(senti_labels)  # [bs, word_emb]
        word_embs = torch.cat((senti_embs.unsqueeze(1), word_embs), dim=1)  # [bs, 1+seq, word_dim]

        out = self.text_cnn(word_embs)  # [bs, fc_feat_dim]
        return out

    def weight_cliping(self, limit=0.01):
        for p in self.parameters():
            p.data.clamp_(-limit, limit)

    def get_optim_criterion(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr), nn.MSELoss()
