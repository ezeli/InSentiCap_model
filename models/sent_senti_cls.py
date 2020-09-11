import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentenceSentimentClassifier(nn.Module):
    def __init__(self, idx2word, sentiment_categories, settings):
        super(SentenceSentimentClassifier, self).__init__()
        self.sentiment_categories = sentiment_categories
        self.pad_id = idx2word.index('<PAD>')
        self.vocab_size = len(idx2word)
        self.word_embed = nn.Sequential(nn.Embedding(self.vocab_size, settings['word_emb_dim'],
                                                     padding_idx=self.pad_id),
                                        nn.ReLU(),
                                        nn.Dropout(settings['dropout_p']))

        rnn_bidirectional = False
        self.rnn = nn.LSTM(settings['word_emb_dim'], settings['rnn_hid_dim'], bidirectional=rnn_bidirectional)
        self.drop = nn.Dropout(settings['dropout_p'])
        if rnn_bidirectional:
            rnn_hid_dim = 2*settings['rnn_hid_dim']
        else:
            rnn_hid_dim = settings['rnn_hid_dim']
        self.excitation = nn.Sequential(
            nn.Linear(rnn_hid_dim, rnn_hid_dim),
            nn.ReLU(),
            nn.Linear(rnn_hid_dim, rnn_hid_dim),
            nn.Sigmoid(),
        )
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.sent_senti_cls = nn.Sequential(
            nn.Linear(rnn_hid_dim, rnn_hid_dim),
            nn.ReLU(),
            nn.Dropout(settings['dropout_p']),
            nn.Linear(rnn_hid_dim, len(sentiment_categories)),
        )

    def forward(self, seqs, lengths):
        seqs = self.word_embed(seqs)  # [bs, max_seq_len, word_dim]
        seqs = pack_padded_sequence(seqs, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(seqs)
        out = pad_packed_sequence(out, batch_first=True)[0]  # [bs, seq_len, rnn_hid]
        out = self.drop(out)

        excitation_res = self.excitation(out)  # [bs, max_len, rnn_hid]
        # excitation_res = self.drop(excitation_res)
        excitation_res = pad_packed_sequence(
            pack_padded_sequence(excitation_res, lengths, batch_first=True, enforce_sorted=False),
            batch_first=True)[0]
        squeeze_res = self.squeeze(excitation_res).permute(0, 2, 1)  # [bs, 1, max_len]
        # squeeze_res = squeeze_res.masked_fill(squeeze_res == 0, -1e10)
        # squeeze_res = squeeze_res.softmax(dim=-1)
        sent_feats = squeeze_res.bmm(out).squeeze(dim=1)  # [bs, rnn_hid]
        pred = self.sent_senti_cls(sent_feats)  # [bs, 3]

        return pred, squeeze_res.squeeze(dim=1)

    def sample(self, seqs, lengths):
        self.eval()
        pred, att_weights = self.forward(seqs, lengths)
        result = []
        result_w = []
        for p in pred:
            res = int(p.argmax(-1))
            result.append(res)
            result_w.append(self.sentiment_categories[res])

        return result, result_w, att_weights

    def get_optim_and_crit(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay), \
               nn.CrossEntropyLoss()
