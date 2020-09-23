from collections import namedtuple
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


BeamCandidate = namedtuple('BeamCandidate',
                           ['state', 'log_prob_sum', 'log_prob_seq', 'last_word_id', 'word_id_seq'])


class ContentAttention(nn.Module):
    def __init__(self, settings):
        super(ContentAttention, self).__init__()
        self.h2att = nn.Linear(settings['rnn_hid_dim'], settings['att_hid_dim'])
        self.att_alpha = nn.Linear(settings['att_hid_dim'], 1)

        self.weights = []

    def _reset_weights(self):
        self.weights = []

    def forward(self, h, att_feats, p_att_feats):
        # The p_att_feats/p_cpt_feats here are already projected
        h_att = self.h2att(h)  # [bs, att_hid]
        h_att = h_att.unsqueeze(1).expand_as(p_att_feats)  # [bs, num_atts, att_hid]
        p_att_feats = p_att_feats + h_att  # [bs, num_atts, att_hid]
        p_att_feats = p_att_feats.tanh()
        p_att_feats = self.att_alpha(p_att_feats).squeeze(-1)  # [bs, num_atts]
        # p_att_feats = p_att_feats.view(-1, att_size)  # [bs, num_atts]
        weight = p_att_feats.softmax(-1)
        self.weights.append(weight)

        att_res = weight.unsqueeze(1).bmm(att_feats).squeeze(1)  # [bs, feat_emb]
        return att_res


class SentiAttention(nn.Module):
    def __init__(self, settings):
        super(SentiAttention, self).__init__()
        self.h2word = nn.Linear(settings['rnn_hid_dim'], settings['att_hid_dim'])
        self.label2word = nn.Linear(settings['word_emb_dim'], settings['att_hid_dim'])
        self.word_alpha = nn.Linear(settings['att_hid_dim'], 1)

        self.weights = []

    def _reset_weights(self):
        self.weights = []

    def forward(self, h, senti_word_feats, p_senti_word_feats, senti_labels):
        h_word = self.h2word(h)  # [bs, att_hid]
        senti_labels_word = self.label2word(senti_labels)  # [bs, att_hid]
        h_word = h_word.unsqueeze(1).expand_as(p_senti_word_feats)  # [bs, num_stmts, att_hid]
        senti_labels_word = senti_labels_word.unsqueeze(1).expand_as(p_senti_word_feats)  # [bs, num_stmts, att_hid]
        p_senti_word_feats = p_senti_word_feats + h_word + senti_labels_word  # [bs, num_stmts, att_hid]
        p_senti_word_feats = p_senti_word_feats.tanh()
        p_senti_word_feats = self.word_alpha(p_senti_word_feats).squeeze(-1)  # [bs, num_stmts]
        weight = p_senti_word_feats.softmax(-1)
        self.weights.append(weight)

        word_res = weight.unsqueeze(1).bmm(senti_word_feats).squeeze(1)  # [bs, word_emb]
        return word_res


class Attention(nn.Module):
    def __init__(self, settings):
        super(Attention, self).__init__()
        self.cont_att = ContentAttention(settings)
        self.senti_att = SentiAttention(settings)

        self.h2att = nn.Linear(settings['rnn_hid_dim'], settings['att_hid_dim'])
        self.cont2att = nn.Linear(settings['feat_emb_dim'], settings['att_hid_dim'])
        self.senti2att = nn.Linear(settings['feat_emb_dim'], settings['att_hid_dim'])
        self.att_alpha = nn.Linear(settings['att_hid_dim'], 1)

        self.weights = []

    def _reset_weights(self):
        self.weights = []
        self.cont_att._reset_weights()
        self.senti_att._reset_weights()

    def _get_weights(self):
        cont_weights = self.cont_att.weights
        if cont_weights:
            cont_weights = torch.cat(cont_weights, dim=1)
        senti_weights = self.senti_att.weights
        if senti_weights:
            senti_weights = torch.cat(senti_weights, dim=1)
        cont_senti_weights = self.weights
        if cont_senti_weights:
            cont_senti_weights = torch.cat(cont_senti_weights, dim=1)
        self._reset_weights()
        return cont_weights, senti_weights, cont_senti_weights

    def forward(self, h, att_feats, p_att_feats, senti_word_feats,
                p_senti_word_feats, senti_labels):
        if att_feats is None:  # for seq2seq
            senti_res = self.senti_att(h, senti_word_feats, p_senti_word_feats, senti_labels)  # [bs, feat_emb]
            return senti_res
        cont_res = self.cont_att(h, att_feats, p_att_feats)  # [bs, feat_emb]
        if senti_word_feats is None:  # for xe
            return cont_res

        # for rl
        senti_res = self.senti_att(h, senti_word_feats, p_senti_word_feats, senti_labels)  # [bs, feat_emb]

        h_att = self.h2att(h)  # [bs, att_hid]
        cont_att = self.cont2att(cont_res)  # [bs, att_hid]
        senti_att = self.senti2att(senti_res)  # [bs, att_hid]
        weight = cont_att + senti_att + h_att  # [bs, att_hid]
        weight = weight.tanh()
        weight = self.att_alpha(weight).sigmoid()  # [bs, 1]
        # weight = (weight > 0.5).type(weight.dtype)
        self.weights.append(weight)

        res = weight * cont_res + (1 - weight) * senti_res
        return res


class Captioner(nn.Module):
    def __init__(self, idx2word, sentiment_categories, settings):
        super(Captioner, self).__init__()
        self.idx2word = idx2word
        self.pad_id = idx2word.index('<PAD>')
        self.unk_id = idx2word.index('<UNK>')
        self.sos_id = idx2word.index('<SOS>') if '<SOS>' in idx2word else self.pad_id
        self.eos_id = idx2word.index('<EOS>') if '<SOS>' in idx2word else self.pad_id
        self.neu_idx = sentiment_categories.index('neutral')

        self.vocab_size = len(idx2word)
        self.drop = nn.Dropout(settings['dropout_p'])
        self.word_embed = nn.Sequential(nn.Embedding(self.vocab_size, settings['word_emb_dim'],
                                                     padding_idx=self.pad_id),
                                        nn.ReLU())
        self.senti_label_embed = nn.Sequential(nn.Embedding(len(sentiment_categories), settings['word_emb_dim']),
                                               nn.ReLU())
        self.fc_embed = nn.Sequential(nn.Linear(settings['fc_feat_dim'], settings['feat_emb_dim']),
                                      nn.ReLU())
        self.cpt2fc = nn.Sequential(nn.Linear(settings['word_emb_dim'], settings['feat_emb_dim']),
                                    nn.ReLU())
        self.att_embed = nn.Sequential(nn.Linear(settings['att_feat_dim'], settings['feat_emb_dim']),
                                       nn.ReLU())
        # self.senti_embed = nn.Sequential(nn.Linear(settings['sentiment_feat_dim'], settings['feat_emb_dim']),
        #                                  nn.LayerNorm(settings['feat_emb_dim']))

        self.att_lstm = nn.LSTMCell(settings['rnn_hid_dim'] + settings['feat_emb_dim'] + settings['word_emb_dim'],
                                    settings['rnn_hid_dim'])  # h^2_t-1, fc, we
        self.att2att = nn.Sequential(nn.Linear(settings['feat_emb_dim'], settings['att_hid_dim']),
                                     nn.ReLU())
        self.senti2att = nn.Sequential(nn.Linear(settings['word_emb_dim'], settings['att_hid_dim']),
                                       nn.ReLU())
        self.attention = Attention(settings)

        # TODO now: word_emb_dim == feat_emb_dim
        # self.senti2feat = nn.Sequential(nn.Linear(settings['word_emb_dim'], settings['feat_emb_dim']),
        #                                 nn.ReLU())
        self.lang_lstm = nn.LSTMCell(settings['rnn_hid_dim'] + settings['feat_emb_dim'],
                                     settings['rnn_hid_dim'])  # \hat v, h^1_t

        self.classifier = nn.Linear(settings['rnn_hid_dim'], self.vocab_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros([2, bsz, self.att_lstm.hidden_size]),  # h_att, h_lang
                weight.new_zeros([2, bsz, self.att_lstm.hidden_size]))  # c_att, c_lang

    def forward_step(self, it, state, fc_feats, att_feats=None, p_att_feats=None,
                     senti_word_feats=None, p_senti_word_feats=None, senti_labels=None):
        xt = self.word_embed(it)
        if senti_labels is not None:
            xt = xt + senti_labels
        prev_h = state[0][1]  # [bs, rnn_hid]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [bs, rnn_hid+feat_emb+word_emb]
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # [bs, rnn_hid]

        att = self.attention(h_att, att_feats, p_att_feats, senti_word_feats,
                             p_senti_word_feats, senti_labels)  # [bs, feat_emb+word_emb]

        lang_lstm_input = torch.cat([att, h_att], 1)  # [bs, feat_emb+rnn_hid]
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # bs*rnn_hid
        output = self.drop(h_lang)  # [bs, rnn_hid]
        logprobs = F.log_softmax(self.classifier(output), dim=1)  # [bs, vocab]

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return logprobs, state

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'xe')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, 'forward_' + mode)(*args, **kwargs)

    def forward_xe(self, fc_feats, att_feats, cpt_words, captions, senti_labels, ss_prob=0.0):
        batch_size = fc_feats.size(0)
        outputs = []

        fc_feats = self.fc_embed(fc_feats)  # [bs, feat_emb]
        self.fc_feats = fc_feats
        fc_feats = self.drop(fc_feats)
        cpt_feats = self.word_embed(cpt_words)  # [bs, num_cpts, word_emb]
        cpt_feats = cpt_feats.mean(dim=1)  # [bs, word_emb]
        cpt_feats = self.cpt2fc(cpt_feats)  # [bs, feat_emb]
        self.cpt_feats = cpt_feats
        # TODO
        # cpt_feats = self.drop(cpt_feats)

        att_feats = att_feats.view(batch_size, -1, att_feats.shape[-1])  # [bs, num_atts, att_feat]
        att_feats = self.att_embed(att_feats)  # [bs, num_atts, feat_emb]
        att_feats = self.drop(att_feats)
        p_att_feats = self.att2att(att_feats)  # [bs, num_atts, att_hid]

        senti_labels = self.senti_label_embed(senti_labels)  # [bs, word_emb]
        senti_labels = self.drop(senti_labels)

        state = self.init_hidden(batch_size)

        for i in range(captions.size(1) - 1):
            if self.training and i >= 1 and ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < ss_prob
                if sample_mask.sum() == 0:
                    it = captions[:, i].clone()  # bs
                else:
                    sample_ind = sample_mask.nonzero(as_tuple=False).view(-1)
                    it = captions[:, i].clone()  # bs
                    prob_prev = outputs[i - 1].detach().exp()  # bs*vocab_size, fetch prev distribution
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = captions[:, i].clone()  # bs

            output, state = self.forward_step(
                it, state, fc_feats, att_feats, p_att_feats, senti_labels=senti_labels)
            outputs.append(output)

        self.cont_weights, self.senti_weights, self.cont_senti_weights = \
            self.attention._get_weights()

        outputs = torch.stack(outputs, dim=1)  # [bs, max_len, vocab_size]
        return outputs

    def forward_seq2seq(self, senti_captions, cpt_words, senti_words, senti_labels,
                        ss_prob=0.0):
        batch_size = senti_captions.size(0)
        outputs = []

        cpt_feats = self.word_embed(cpt_words)  # [bs, num_cpts, word_emb]
        cpt_feats = cpt_feats.mean(dim=1)  # [bs, word_emb]
        cpt_feats = self.cpt2fc(cpt_feats)  # [bs, feat_emb]
        cpt_feats = self.drop(cpt_feats)
        fc_feats = cpt_feats

        senti_words = torch.cat(
            [senti_words.new_zeros(batch_size, 1).fill_(self.pad_id), senti_words],
            dim=1)    # [bs, num_stmts]
        senti_word_feats = self.word_embed(senti_words)  # [bs, num_stmts, word_emb]
        senti_word_feats = self.drop(senti_word_feats)
        p_senti_word_feats = self.senti2att(senti_word_feats)  # [bs, num_stmts, att_hid]

        senti_labels = self.senti_label_embed(senti_labels)  # [bs, word_emb]
        senti_labels = self.drop(senti_labels)

        state = self.init_hidden(batch_size)

        for i in range(senti_captions.size(1) - 1):
            if self.training and i >= 1 and ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < ss_prob
                if sample_mask.sum() == 0:
                    it = senti_captions[:, i].clone()  # bs
                else:
                    sample_ind = sample_mask.nonzero(as_tuple=False).view(-1)
                    it = senti_captions[:, i].clone()  # bs
                    prob_prev = outputs[i - 1].detach().exp()  # bs*vocab_size, fetch prev distribution
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = senti_captions[:, i].clone()  # bs

            output, state = self.forward_step(
                it, state, fc_feats, senti_word_feats=senti_word_feats,
                p_senti_word_feats=p_senti_word_feats, senti_labels=senti_labels)
            outputs.append(output)

        self.cont_weights, self.senti_weights, self.cont_senti_weights = \
            self.attention._get_weights()

        outputs = torch.stack(outputs, dim=1)  # [bs, max_len, vocab_size]
        return outputs

    def forward_rl(self, fc_feats, att_feats, cpt_words, senti_words, senti_labels,
                   max_seq_len, sample_max):
        batch_size = fc_feats.shape[0]

        fc_feats = self.fc_embed(fc_feats)  # [bs, feat_emb]
        self.fc_feats = fc_feats
        fc_feats = self.drop(fc_feats)
        cpt_feats = self.word_embed(cpt_words)  # [bs, num_cpts, word_emb]
        cpt_feats = cpt_feats.mean(dim=1)  # [bs, word_emb]
        cpt_feats = self.cpt2fc(cpt_feats)  # [bs, feat_emb]
        self.cpt_feats = cpt_feats

        att_feats = att_feats.view(batch_size, -1, att_feats.shape[-1])  # [bs, num_atts, att_feat]
        att_feats = self.att_embed(att_feats)  # [bs, num_atts, feat_emb]
        att_feats = self.drop(att_feats)
        p_att_feats = self.att2att(att_feats)  # [bs, num_atts, att_hid]

        senti_words = torch.cat(
            [senti_words.new_zeros(batch_size, 1).fill_(self.pad_id), senti_words],
            dim=1)  # [bs, num_stmts]
        senti_word_feats = self.word_embed(senti_words)  # [bs, num_stmts, word_emb]
        senti_word_feats = self.drop(senti_word_feats)
        p_senti_word_feats = self.senti2att(senti_word_feats)  # [bs, num_stmts, att_hid]

        senti_labels = self.senti_label_embed(senti_labels)  # [bs, word_emb]
        senti_labels = self.drop(senti_labels)

        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros((batch_size, max_seq_len), dtype=torch.long)
        seq_logprobs = fc_feats.new_zeros((batch_size, max_seq_len))
        seq_masks = fc_feats.new_zeros((batch_size, max_seq_len))
        it = fc_feats.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)  # first input <SOS>
        unfinished = it == self.sos_id
        for t in range(max_seq_len):
            logprobs, state = self.forward_step(
                it, state, fc_feats, att_feats, p_att_feats,
                senti_word_feats, p_senti_word_feats, senti_labels)

            if sample_max:
                sample_logprobs, it = torch.max(logprobs, 1)
            else:
                prob_prev = torch.exp(logprobs)
                it = torch.multinomial(prob_prev, 1)
                sample_logprobs = logprobs.gather(1, it)  # gather the logprobs at sampled positions
            it = it.view(-1).long()
            sample_logprobs = sample_logprobs.view(-1)

            seq_masks[:, t] = unfinished
            it = it * unfinished.type_as(it)  # bs
            seq[:, t] = it
            seq_logprobs[:, t] = sample_logprobs

            unfinished = unfinished * (it != self.eos_id)
            if unfinished.sum() == 0:
                break

        self.cont_weights, self.senti_weights, self.cont_senti_weights = \
            self.attention._get_weights()

        return seq, seq_logprobs, seq_masks

    def sample(self, fc_feat, att_feat, senti_words=None, senti_label=None,
               beam_size=3, decoding_constraint=1, max_seq_len=16):
        self.eval()
        fc_feats = fc_feat.view(1, -1)  # [1, fc_feat]
        att_feats = att_feat.view(1, -1, att_feat.shape[-1])  # [1, num_atts, att_feat]

        fc_feats = self.fc_embed(fc_feats)  # [bs, feat_emb]
        fc_feats = self.drop(fc_feats)

        att_feats = self.att_embed(att_feats)  # [bs, num_atts, feat_emb]
        att_feats = self.drop(att_feats)
        p_att_feats = self.att2att(att_feats)  # [bs, num_atts, att_hid]

        if senti_words is not None:
            senti_words = senti_words.view(1, -1)
            senti_words = torch.cat(
                [senti_words.new_zeros(1, 1).fill_(self.pad_id), senti_words],
                dim=1)  # [bs, num_stmts]
            senti_word_feats = self.word_embed(senti_words)  # [bs, num_stmts, word_emb]
            senti_word_feats = self.drop(senti_word_feats)
            p_senti_word_feats = self.senti2att(senti_word_feats)  # [bs, num_stmts, att_hid]

            senti_labels = self.senti_label_embed(senti_label)  # [bs, word_emb]
            senti_labels = self.drop(senti_labels)
        else:
            senti_word_feats = p_senti_word_feats = senti_labels = None

        state = self.init_hidden(1)
        candidates = [BeamCandidate(state, 0., [], self.sos_id, [])]
        for t in range(max_seq_len):
            tmp_candidates = []
            end_flag = True
            for candidate in candidates:
                state, log_prob_sum, log_prob_seq, last_word_id, word_id_seq = candidate
                if t > 0 and last_word_id == self.eos_id:
                    tmp_candidates.append(candidate)
                else:
                    end_flag = False
                    it = fc_feats.type(torch.long).new_tensor([last_word_id])
                    logprobs, state = self.forward_step(
                        it, state, fc_feats, att_feats, p_att_feats,
                        senti_word_feats, p_senti_word_feats, senti_labels)  # [1, vocab_size]
                    logprobs = logprobs.squeeze(0)  # vocab_size
                    if self.pad_id != self.eos_id:
                        logprobs[self.pad_id] = float('-inf')  # do not generate <PAD> and <SOS>
                        logprobs[self.sos_id] = float('-inf')
                        logprobs[self.unk_id] = float('-inf')
                    if decoding_constraint:  # do not generate last step word
                        logprobs[last_word_id] = float('-inf')

                    output_sorted, index_sorted = torch.sort(logprobs, descending=True)
                    for k in range(beam_size):
                        log_prob, word_id = output_sorted[k], index_sorted[k]  # tensor, tensor
                        log_prob = float(log_prob)
                        word_id = int(word_id)
                        tmp_candidates.append(BeamCandidate(state, log_prob_sum + log_prob,
                                                            log_prob_seq + [log_prob],
                                                            word_id, word_id_seq + [word_id]))
            candidates = sorted(tmp_candidates, key=lambda x: x.log_prob_sum, reverse=True)[:beam_size]
            if end_flag:
                break

        self.cont_weights, self.senti_weights, self.cont_senti_weights = \
            self.attention._get_weights()

        # captions, scores
        captions = [' '.join([self.idx2word[idx] for idx in candidate.word_id_seq if idx != self.eos_id])
                    for candidate in candidates]
        scores = [candidate.log_prob_sum for candidate in candidates]
        return captions, scores

    def get_optim_criterion(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),\
               XECriterion(), nn.MSELoss()  # xe, domain align


class XECriterion(nn.Module):
    def __init__(self):
        super(XECriterion, self).__init__()

    def forward(self, pred, target, lengths):
        max_len = max(lengths)
        mask = pred.new_zeros(len(lengths), max_len)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1

        loss = - pred.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        loss = torch.sum(loss) / torch.sum(mask)

        return loss
