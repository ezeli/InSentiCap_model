import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BackTranslator(nn.Module):
    def __init__(self, num_senti, settings):
        super(BackTranslator, self).__init__()
        hid_dim = settings['rnn_hid_dim']
        # assert hid_dim % 2 == 0

        self.senti_embed = nn.Sequential(
            nn.Embedding(num_senti, settings['word_emb_dim']),
            nn.ReLU())
        self.emb_drop = nn.Dropout(settings['dropout_p'])
        # gru TODO: bidirectional=False
        self.gru = nn.GRU(settings['word_emb_dim'], hid_dim,
                          batch_first=True, bidirectional=True)
        self.gru_hid_drop = nn.Dropout(settings['dropout_p'])
        self.output = nn.Linear(hid_dim, settings['fc_feat_dim'])

    def forward(self, word_embs, lengths, senti_labels):
        # [bs, seq, word_dim], [bs]
        senti_embs = self.senti_embed(senti_labels)  # [bs, word_emb]
        word_embs = torch.cat((senti_embs.unsqueeze(1), word_embs), dim=1)  # [bs, 1+seq, word_dim]
        word_embs = self.emb_drop(word_embs)
        lengths += 1
        word_embs = pack_padded_sequence(word_embs, lengths,
                                         batch_first=True, enforce_sorted=False)
        _, ht = self.gru(word_embs)
        ht = ht[0] + ht[1]  # [bs, hid_dim]
        # ht = torch.cat((ht[0], ht[1]), dim=-1)  # [bs, hid_dim]
        ht = self.gru_hid_drop(ht)
        return self.output(ht)  # [bs, fc_feat_dim]

        # senti_embs = self.senti_embed(senti_labels)  # [bs, word_emb]
        # senti_embs = senti_embs.unsqueeze(1).expand_as(word_embs)  # [bs, seq, word_emb]
        # word_embs = torch.cat((word_embs, senti_embs), dim=2)  # [bs, seq, 2*word_emb]
        # out, ht = self.gru(self.emb_drop(word_embs))
        # out = self.gru_out_drop(out)  # [bs, seq, hid_dim]
        # ht = self.gru_hid_drop(ht)  # [2, bs, hid_dim/2]
        # return out, ht

    def get_optim_criterion(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr), nn.MSELoss()
