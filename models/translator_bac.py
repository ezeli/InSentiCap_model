import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class CaptionEncoder(nn.Module):
    def __init__(self, num_senti, settings):
        super(CaptionEncoder, self).__init__()
        hid_dim = settings['rnn_hid_dim']
        assert hid_dim % 2 == 0

        self.senti_embed = nn.Sequential(
            nn.Embedding(num_senti, settings['word_emb_dim']),
            nn.ReLU())
        self.emb_drop = nn.Dropout(0.2)
        # gru
        self.gru = nn.GRU(2*settings['word_emb_dim'], hid_dim / 2, batch_first=True, bidirectional=True)
        self.gru_out_drop = nn.Dropout(0.2)
        self.gru_hid_drop = nn.Dropout(0.3)

    def forward(self, word_embs, senti_labels):
        # [bs, seq, word_dim], [bs]
        senti_embs = self.senti_embed(senti_labels)  # [bs, word_emb]
        senti_embs = senti_embs.unsqueeze(1).expand_as(word_embs)  # [bs, seq, word_emb]
        word_embs = torch.cat((word_embs, senti_embs), dim=2)  # [bs, seq, 2*word_emb]
        out, ht = self.gru(self.emb_drop(word_embs))
        out = self.gru_out_drop(out)  # [bs, seq, hid_dim]
        ht = self.gru_hid_drop(ht)  # [2, bs, hid_dim/2]
        return out, ht


class CaptionDecoder(nn.Module):
    def __init__(self, word_embed, settings):
        super(CaptionDecoder, self).__init__()
        self.word_embed = word_embed
        self.vocab_size = word_embed[0].weight.size(0)
        hid_dim = settings['rnn_hid_dim']
        # gru
        self.gru = nn.GRU(settings['word_emb_dim'], hid_dim, batch_first=True)
        self.gru_drop = nn.Dropout(0.2)
        # attention
        self.att_mlp = nn.Linear(hid_dim, hid_dim, bias=False)
        self.attn_softmax = nn.Softmax(dim=-1)
        # output
        self.cla = nn.Linear(hid_dim*2, self.vocab_size)

    def forward(self, captions, lengths, hidden, encoder_outs):
        # captions:  [batch, seq]
        emb = self.word_embed(captions)  # [bs, max_len, word_emb]
        out, hidden = self.gru(emb, hidden)  # [bs, max_len, hid_dim]

        out_proj = self.att_mlp(out)  # [bs, max_len, hid_dim]
        enc_out_perm = encoder_outs.permute(0, 2, 1)  # [bs, hid_dim, y]
        e_exp = torch.bmm(out_proj, enc_out_perm)  # [bs, max_len, y]
        attn = self.attn_softmax(e_exp)  # [bs, max_len, y]

        ctx = torch.bmm(attn, encoder_outs)  # [bs, max_len, hid_dim]
        full_ctx = torch.cat([self.gru_drop(out), ctx], dim=-1)  # [bs, max_len, hid_dim*2]

        out = self.cla(full_ctx)  # [bs, max_len, vocab]
        out = pack_padded_sequence(out, lengths, batch_first=True)[0]
        return out, hidden, attn


class BackTranslator(nn.Module):
    def __init__(self, word_embed, num_senti, settings):
        super(BackTranslator, self).__init__()
        self.enc = CaptionEncoder(num_senti, settings)
        self.dec = CaptionDecoder(word_embed, settings)

    def forward(self, enc_word_embs, senti_labels, captions, lengths):
        # enc_word_embs: [bs, seq, word_dim], senti_labels: [bs], captions: [bs, seq]
        out_enc, hid_enc = self.enc(enc_word_embs, senti_labels)
        hid_enc = torch.cat([hid_enc[0, :, :], hid_enc[1, :, :]], dim=1).\
            unsqueeze(0)  # [1, bs, hidden]
        return self.dec(captions, lengths, hid_enc, out_enc)

    def get_optim_criterion(self, lr):
        return torch.optim.Adam(self.parameters(), lr=lr), nn.CrossEntropyLoss()
