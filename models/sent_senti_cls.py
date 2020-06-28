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
        self.amp = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Dropout(settings['dropout_p'])
        )
        num_senti = len(sentiment_categories)
        if settings['rnn_bidirectional']:
            self.classifier = nn.Linear(2*settings['rnn_hid_dim'], num_senti)
        else:
            self.classifier = nn.Linear(settings['rnn_hid_dim'], num_senti)

    def forward(self, seqs, lengths):
        seqs = self.word_embed(seqs)  # [bs, max_seq_len, word_dim]
        seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
        out, _ = self.rnn(seqs)
        out = pad_packed_sequence(out, batch_first=True)[0]  # [bs, seq_len, rnn_hid]
        out = self.amp(out.permute(0, 2, 1)).view(out.size(0), -1)  # [bs, rnn_hid]
        pred = self.classifier(out)  # [bs, num_senti]
        return pred

    def sample(self, seqs, lengths):
        self.eval()
        pred = self.forward(seqs, lengths)
        result = []
        result_w = []
        for p in pred:
            res = int(p.argmax(-1))
            result.append(res)
            result_w.append(self.sentiment_categories[res])

        return result, result_w

    def get_optim_and_crit(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay), \
               nn.CrossEntropyLoss()
