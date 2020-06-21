from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F


BeamCandidate = namedtuple('BeamCandidate',
                           ['state', 'log_prob_sum', 'log_prob_seq', 'last_word_id', 'word_id_seq'])


class ContentAttention(nn.Module):
    def __init__(self, settings):
        super(ContentAttention, self).__init__()
        self.att_hid_dim = settings['att_hid_dim']

        self.h2att = nn.Linear(settings['rnn_hid_dim'], self.att_hid_dim)
        self.h2cpt = nn.Linear(settings['rnn_hid_dim'], self.att_hid_dim)
        self.att_alpha = nn.Linear(self.att_hid_dim, 1)
        self.cpt_alpha = nn.Linear(self.att_hid_dim, 1)

    def forward(self, h, att_feats, cpt_feats, p_att_feats, p_cpt_feats):
        # The p_att_feats/p_cpt_feats here are already projected
        h_att = self.h2att(h)  # [bs, att_hid]
        h_att = h_att.unsqueeze(1).expand_as(p_att_feats)  # [bs, num_atts, att_hid]
        p_att_feats = p_att_feats + h_att  # [bs, num_atts, att_hid]
        p_att_feats = p_att_feats.tanh()
        p_att_feats = self.att_alpha(p_att_feats).squeeze(-1)  # [bs, num_atts]
        # p_att_feats = p_att_feats.view(-1, att_size)  # [bs, num_atts]
        weight = p_att_feats.softmax(-1)
        att_res = weight.unsqueeze(1).bmm(att_feats).squeeze(1)  # [bs, feat_emb]

        h_cpt = self.h2cpt(h)  # [bs, att_hid]
        h_cpt = h_cpt.unsqueeze(1).expand_as(p_cpt_feats)  # [bs, num_cpts, att_hid]
        p_cpt_feats = p_cpt_feats + h_cpt  # [bs, num_cpts, att_hid]
        p_cpt_feats = p_cpt_feats.tanh()
        p_cpt_feats = self.cpt_alpha(p_cpt_feats).squeeze(-1)  # [bs, num_cpts]
        weight = p_cpt_feats.softmax(-1)
        cpt_res = weight.unsqueeze(1).bmm(cpt_feats).squeeze(1)  # [bs, word_emb]

        # TODO: use add op replace cat op
        res = torch.cat([att_res, cpt_res], dim=-1)  # [bs, feat_emb+word_emb]
        return res


class SentiAttention(nn.Module):
    def __init__(self, settings):
        super(SentiAttention, self).__init__()
        self.att_hid_dim = settings['att_hid_dim']

        self.h2word = nn.Linear(settings['rnn_hid_dim'], self.att_hid_dim)
        self.word_alpha = nn.Linear(self.att_hid_dim, 1)

    def forward(self, h, senti_feats, senti_word_feats, p_senti_word_feats):
        h_word = self.h2word(h)  # [bs, att_hid]
        h_word = h_word.unsqueeze(1).expand_as(p_senti_word_feats)  # [bs, num_stmts, att_hid]
        p_senti_word_feats = p_senti_word_feats + h_word  # [bs, num_stmts, att_hid]
        p_senti_word_feats = p_senti_word_feats.tanh()
        p_senti_word_feats = self.word_alpha(p_senti_word_feats).squeeze(-1)  # [bs, num_stmts]
        weight = p_senti_word_feats.softmax(-1)
        word_res = weight.unsqueeze(1).bmm(senti_word_feats).squeeze(1)  # [bs, word_emb]

        # TODO: use add op replace cat op
        res = torch.cat([senti_feats, word_res], dim=-1)  # [bs, feat_emb+word_emb]
        return res


class Attention(nn.Module):
    def __init__(self, settings):
        super(Attention, self).__init__()
        self.cont_att = ContentAttention(settings)
        self.senti_att = SentiAttention(settings)

        self.h2att = nn.Linear(settings['rnn_hid_dim'], settings['att_hid_dim'])
        self.cont2att = nn.Linear(settings['feat_emb_dim'] + settings['word_emb_dim'],
                                  settings['att_hid_dim'])
        self.senti2att = nn.Linear(settings['feat_emb_dim'] + settings['word_emb_dim'],
                                   settings['att_hid_dim'])
        self.att_alpha = nn.Linear(settings['att_hid_dim'], 1)

    def forward(self, h, att_feats, cpt_feats, p_att_feats, p_cpt_feats,
                senti_feats=None, senti_word_feats=None, p_senti_word_feats=None):
        cont_res = self.cont_att(h, att_feats, cpt_feats, p_att_feats, p_cpt_feats)  # [bs, feat_emb+word_emb]
        if senti_feats is None:
            return cont_res
        senti_res = self.senti_att(h, senti_feats, senti_word_feats, p_senti_word_feats)  # [bs, feat_emb+word_emb]

        h_att = self.h2att(h)  # [bs, att_hid]
        cont_att = self.cont2att(cont_res)  # [bs, att_hid]
        senti_att = self.senti2att(senti_res)  # [bs, att_hid]
        weight = cont_att + senti_att + h_att  # [bs, att_hid]
        weight = weight.tanh()
        weight = self.att_alpha(weight).sigmoid()  # [bs, 1]

        res = weight * cont_res + (1 - weight) * senti_res
        return res


class Captioner(nn.Module):
    def __init__(self, idx2word, settings):
        super(Captioner, self).__init__()
        self.idx2word = idx2word
        self.pad_id = idx2word.index('<PAD>')
        self.unk_id = idx2word.index('<UNK>')
        self.sos_id = idx2word.index('<SOS>') if '<SOS>' in idx2word else self.pad_id
        self.eos_id = idx2word.index('<EOS>') if '<SOS>' in idx2word else self.pad_id
        self.dropout_p = settings['dropout_p']

        self.vocab_size = len(idx2word)
        self.word_embed = nn.Sequential(nn.Embedding(self.vocab_size, settings['word_emb_dim'],
                                                     padding_idx=self.pad_id),
                                        nn.ReLU(),
                                        nn.Dropout(settings['dropout_p']))
        # self.fc_embed = nn.Linear(settings['fc_feat_dim'], settings['feat_emb_dim'])
        self.fc_embed = nn.Sequential(nn.Linear(settings['fc_feat_dim'], settings['feat_emb_dim']),
                                      nn.ReLU(),
                                      nn.Dropout(settings['dropout_p']))
        self.att_embed = nn.Sequential(nn.Linear(settings['att_feat_dim'], settings['feat_emb_dim']),
                                       nn.ReLU(),
                                       nn.Dropout(settings['dropout_p']))
        self.senti_embed = nn.Linear(settings['sentiment_feat_dim'], settings['feat_emb_dim'])
        # TODO
        # self.cpt_embed = nn.Sequential(nn.Linear(settings['word_emb_dim'], settings['feat_emb_dim']),
        #                                nn.ReLU(),
        #                                nn.Dropout(settings['dropout_p']))

        self.att_lstm = nn.LSTMCell(settings['rnn_hid_dim'] + settings['feat_emb_dim'] + settings['word_emb_dim'],
                                    settings['rnn_hid_dim'])  # h^2_t-1, fc, we
        self.att2att = nn.Linear(settings['feat_emb_dim'], settings['att_hid_dim'])
        self.cpt2att = nn.Linear(settings['word_emb_dim'], settings['att_hid_dim'])
        self.senti2att = nn.Linear(settings['word_emb_dim'], settings['att_hid_dim'])
        self.attention = Attention(settings)
        self.lang_lstm = nn.LSTMCell(settings['rnn_hid_dim'] + settings['feat_emb_dim'] + settings['word_emb_dim'],
                                     settings['rnn_hid_dim'])  # \hat v, h^1_t
        self.lang_drop = nn.Dropout(settings['dropout_p'])

        self.classifier = nn.Linear(settings['rnn_hid_dim'], self.vocab_size)

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros([2, bsz, self.att_lstm.hidden_size]),  # h_att, h_lang
                weight.new_zeros([2, bsz, self.att_lstm.hidden_size]))  # c_att, c_lang

    def forward_step(self, it, fc_feats, att_feats, cpt_feats,
                     p_att_feats, p_cpt_feats, state, senti_feats=None,
                     senti_word_feats=None, p_senti_word_feats=None):
        xt = self.word_embed(it)
        prev_h = state[0][1]  # [bs, rnn_hid]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt], 1)  # [bs, rnn_hid+feat_emb+word_emb]
        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))  # [bs, rnn_hid]

        att = self.attention(h_att, att_feats, cpt_feats, p_att_feats, p_cpt_feats,
                             senti_feats, senti_word_feats, p_senti_word_feats)  # [bs, feat_emb+word_emb]

        lang_lstm_input = torch.cat([att, h_att], 1)  # [bs, rnn_hid+feat_emb+word_emb]
        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))  # bs*rnn_hid
        output = self.lang_drop(h_lang)  # [bs, rnn_hid]
        logprobs = F.log_softmax(self.classifier(output), dim=1)

        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))
        return logprobs, state

    def forward(self, *args, **kwargs):
        mode = kwargs.get('mode', 'xe')
        if 'mode' in kwargs:
            del kwargs['mode']
        return getattr(self, 'forward_' + mode)(*args, **kwargs)

    def forward_xe(self, fc_feats, att_feats, concepts, captions, ss_prob=0.0):
        batch_size = fc_feats.size(0)
        outputs = []
        # outputs = fc_feats.new_zeros(batch_size, lengths[0], self.vocab_size)

        fc_feats = self.fc_embed(fc_feats)  # [bs, feat_emb]
        att_feats = att_feats.view(batch_size, -1, att_feats.shape[-1])  # [bs, num_atts, att_feat]
        att_feats = self.att_embed(att_feats)  # [bs, num_atts, feat_emb]
        cpt_feats = self.word_embed(concepts)  # [bs, num_cpts, word_emb]
        # p_att_feats/p_cpt_feats are used for attention, we cache it in advance to reduce computation cost
        p_att_feats = self.att2att(att_feats)  # [bs, num_atts, att_hid]
        p_cpt_feats = self.cpt2att(cpt_feats)  # [bs, num_cpts, att_hid]
        state = self.init_hidden(batch_size)

        for i in range(captions.size(1) - 1):
            if self.training and i >= 1 and ss_prob > 0.0:  # otherwise no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < ss_prob
                if sample_mask.sum() == 0:
                    it = captions[:, i].clone()  # bs
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = captions[:, i].clone()  # bs
                    prob_prev = outputs[i - 1].detach().softmax(dim=-1)  # bs*vocab_size, fetch prev distribution
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = captions[:, i].clone()  # bs

            output, state = self.forward_step(
                it, fc_feats, att_feats, cpt_feats, p_att_feats,
                p_cpt_feats, state)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)  # [bs, max_len, vocab_size]
        # outputs = pack_padded_sequence(outputs, lengths, batch_first=True)[0]
        return outputs

    def forward_rl(self, fc_feats, att_feats, concepts, max_seq_len, sample_max):
        batch_size = fc_feats.shape[0]

        fc_feats = self.fc_embed(fc_feats)  # [bs, feat_emb]
        att_feats = att_feats.view(batch_size, -1, att_feats.shape[-1])  # [bs, num_atts, att_feat]
        att_feats = self.att_embed(att_feats)  # [bs, num_atts, feat_emb]
        cpt_feats = self.word_embed(concepts)  # [bs, num_cpts, word_emb]
        p_att_feats = self.att2att(att_feats)  # [bs, num_atts, att_hid]
        p_cpt_feats = self.cpt2att(cpt_feats)  # [bs, num_cpts, att_hid]

        state = self.init_hidden(batch_size)
        seq = fc_feats.new_zeros((batch_size, max_seq_len), dtype=torch.long)
        seq_logprobs = fc_feats.new_zeros((batch_size, max_seq_len))
        seq_masks = fc_feats.new_zeros((batch_size, max_seq_len))
        it = fc_feats.new_zeros(batch_size, dtype=torch.long).fill_(self.sos_id)  # first input <SOS>
        unfinished = it == self.sos_id
        for t in range(max_seq_len):
            logprobs, state = self.forward_step(it, fc_feats, att_feats, cpt_feats,
                                              p_att_feats, p_cpt_feats, state)
                                              # senti_feats, senti_word_feats,
                                              # p_senti_word_feats)

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

        return seq, seq_logprobs, seq_masks

    def sample(self, fc_feat, att_feat, concepts,
               senti_feat=None, sentiments=None,
               beam_size=3, decoding_constraint=1, max_seq_len=16):
        self.eval()
        fc_feat = fc_feat.view(1, -1)  # [1, fc_feat]
        att_feat = att_feat.view(1, -1, att_feat.shape[-1])  # [1, num_atts, att_feat]
        concepts = concepts.view(1, -1)  # [1, num_cpts]
        fc_feat = self.fc_embed(fc_feat)  # [1, feat_emb]
        att_feat = self.att_embed(att_feat)  # [1, num_atts, feat_emb]
        cpt_feat = self.word_embed(concepts)  # [1, num_cpts, word_emb]
        p_att_feat = self.att2att(att_feat)  # [1, num_atts, att_hid]
        p_cpt_feat = self.cpt2att(cpt_feat)  # [bs, num_cpts, att_hid]

        if senti_feat is not None:
            senti_feat = senti_feat.view(1, -1)  # [1, senti_feat]
            senti_feat = self.senti_embed(senti_feat)  # [1, feat_emb]
            sentiments = sentiments.view(1, -1)  # [1, num_stmts]
            senti_word_feat = self.word_embed(sentiments)  # [1, num_stmts, word_emb]
            p_senti_word_feat = self.senti2att(senti_word_feat)  # [1, num_stmts, att_hid]
        else:
            senti_word_feat = p_senti_word_feat = None

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
                    it = concepts.new_tensor([last_word_id])
                    logprobs, state = self.\
                        forward_step(it, fc_feat, att_feat, cpt_feat,
                                     p_att_feat, p_cpt_feat, state,
                                     senti_feat, senti_word_feat,
                                     p_senti_word_feat)  # [1, vocab_size]
                    logprobs = logprobs.squeeze(0)  # vocab_size
                    # if self.pad_id != self.eos_id:
                    #     output[self.pad_id] = float('-inf')  # do not generate <PAD> and <SOS>
                    #     output[self.sos_id] = float('-inf')
                    #     output[self.unk_id] = float('-inf')
                    # if decoding_constraint:  # do not generate last step word
                    #     output[last_word_id] = float('-inf')

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

        # captions, scores
        captions = [' '.join([self.idx2word[idx] for idx in candidate.word_id_seq if idx != self.eos_id])
                    for candidate in candidates]
        scores = [candidate.log_prob_sum for candidate in candidates]
        return captions, scores

    def get_optim_criterion(self, lr, weight_decay=0):
        return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay),\
               XECriterion()


class XECriterion(nn.Module):
    def __init__(self):
        super(XECriterion, self).__init__()

    def forward(self, pred, target, lengths):
        max_len = max(lengths)
        # pred = pred[:, :max_len]
        # target = target[:, :max_len]
        mask = pred.new_zeros(len(lengths), max_len)
        for i, l in enumerate(lengths):
            mask[i, :l] = 1
        # pred = pred.log_softmax(dim=-1)

        loss = - pred.gather(2, target.unsqueeze(2)).squeeze(2) * mask
        loss = torch.sum(loss) / torch.sum(mask)

        return loss
