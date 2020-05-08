# encoding:utf-8
import random
import tqdm
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .captioner import Captioner
from .discriminator import CaptionDiscriminator
from .classifier import SentimentClassifier
from .translator import BackTranslator
from .sentiment_detector import SentimentDetector


class InSentiCap(nn.Module):
    def __init__(self, idx2word, max_seq_length, sentiment_categories, lrs,
                 hyperparams, real_captions, settings):
        super(InSentiCap, self).__init__()
        self.idx2word = idx2word
        self.pad_id = idx2word.index('<PAD>')
        self.max_seq_length = max_seq_length

        self.captioner = Captioner(idx2word, settings)
        self.senti_detector = SentimentDetector(sentiment_categories, settings)
        self.discriminator = CaptionDiscriminator(settings)
        num_senti = len(sentiment_categories)
        self.classifier = SentimentClassifier(num_senti, settings)
        self.translator = BackTranslator(num_senti, settings)

        self.cap_optim, _ = self.captioner.get_optim_criterion(lrs['cap_lr'])
        self.senti_optim, self.senti_crit = self.senti_detector.get_optim_criterion(lrs['senti_lr'])
        self.dis_optim, self.dis_crit = self.discriminator.get_optim_criterion(lrs['dis_lr'])
        self.cls_optim, self.cls_crit = self.classifier.get_optim_criterion(lrs['cla_lr'])
        self.tra_optim, self.tra_crit = self.translator.get_optim_criterion(lrs['tra_lr'])

        self.hp_dis, self.hp_cls, self.hp_tra = \
            hyperparams['hp_dis'], hyperparams['hp_cls'], hyperparams['hp_tra']

        self.real_captions = real_captions  # {'fact': [[2,44,...],...], 'senti': [(0, [2, 44]),..]}

    def get_batch_real_caps(self, caps_type, batch_size, max_len):
        device = next(self.parameters()).device
        caps = random.sample(self.real_captions[caps_type], batch_size)
        if caps_type == 'senti':
            sentis, caps = zip(*caps)

        lengths = [min(len(c), max_len) for c in caps]
        cap_tensor = torch.LongTensor(len(caps), max(lengths)).to(device).fill_(self.pad_id)
        for i, c in enumerate(caps):
            end_cap = lengths[i]
            cap_tensor[i, :end_cap] = torch.LongTensor(c[:end_cap])
        cap_tensor = self.captioner.word_embed(cap_tensor)  # [bs, max_len, word_dim]
        if caps_type == 'senti':
            sentis = torch.LongTensor(sentis).to(device)
            return (cap_tensor, lengths), sentis
        else:
            return cap_tensor, lengths

    def forward(self, data, data_type, training):
        self.train(training)
        all_losses = torch.FloatTensor(5).fill_(0)
        device = next(self.parameters()).device
        for data_item in tqdm.tqdm(data):
            if data_type == 'fact':
                _, fc_feats, att_feats, cpts_tensor, sentis_tensor = data_item
                senti_labels = None
            elif data_type == 'senti':
                _, fc_feats, att_feats, cpts_tensor, sentis_tensor, senti_labels = data_item
                senti_labels = senti_labels.to(device)
            else:
                raise Exception('data_type(%s) is wrong!' % data_type)

            bs = fc_feats.size(0)
            fc_feats = fc_feats.to(device)
            att_feats = att_feats.to(device)
            cpts_tensor = cpts_tensor.to(device)
            sentis_tensor = sentis_tensor.to(device)
            del data_item

            det_sentis, det_senti_features = self.senti_detector(att_feats)  # [bs, num_sentis], [bs, 14, 14]
            if data_type == 'fact':
                senti_labels = det_sentis.argmax(-1).detach()  # bs
                s_loss = 0
            else:
                s_loss = self.senti_crit(det_sentis, senti_labels)
            cap_out, lengths = self.captioner(
                fc_feats, att_feats, cpts_tensor, det_senti_features,
                sentis_tensor, self.max_seq_length, mode='ft')

            d_real_labels = torch.ones(bs).to(device)
            d_fake_labels = torch.zeros(bs).to(device)
            if training:  # We train D for 5 times more than G
                for _ in range(4):
                    real_caps, _ = self.get_batch_real_caps(
                        'fact', bs, self.max_seq_length)  # [bs, max_len, word_dim]
                    d_real_out = self.discriminator(real_caps)  # [bs]
                    d_real_loss = self.dis_crit(d_real_out, d_real_labels)
                    d_fake_out = self.discriminator(cap_out)
                    d_fake_loss = self.dis_crit(d_fake_out, d_fake_labels)
                    d_loss = d_real_loss + d_fake_loss
                    self.dis_optim.zero_grad()
                    d_loss.backward(retain_graph=True)
                    self.dis_optim.step()
                    self.discriminator.weight_cliping()

                    (real_caps, _), real_senti_labels = self.get_batch_real_caps(
                        'senti', bs, self.max_seq_length)  # [bs, max_len, word_dim], [bs]
                    c_real_out = self.classifier(real_caps)  # [bs, num_senti]
                    c_real_loss = self.cls_crit(c_real_out, real_senti_labels)
                    self.cls_optim.zero_grad()
                    c_real_loss.backward(retain_graph=True)
                    self.cls_optim.step()
                    self.classifier.weight_cliping()

            real_caps, _ = self.get_batch_real_caps(
                'fact', bs, self.max_seq_length)  # [bs, max_len, word_dim]
            d_real_out = self.discriminator(real_caps)  # [bs]
            d_real_loss = self.dis_crit(d_real_out, d_real_labels)
            d_fake_out = self.discriminator(cap_out)
            d_fake_loss = self.dis_crit(d_fake_out, d_fake_labels)
            d_loss = d_real_loss + d_fake_loss

            (real_caps, _), real_senti_labels = self.get_batch_real_caps(
                'senti', bs, self.max_seq_length)  # [bs, max_len, word_dim], [bs]
            c_real_out = self.classifier(real_caps)  # [bs, num_senti]
            c_real_loss = self.cls_crit(c_real_out, real_senti_labels)
            c_fake_out = self.classifier(cap_out)
            c_fake_loss = self.cls_crit(c_fake_out, senti_labels)
            c_loss = c_real_loss + c_fake_loss

            t_out = self.translator(cap_out, senti_labels)  # [bs, fc_feat_dim]
            t_loss = self.tra_crit(t_out, fc_feats)

            cap_loss = -self.hp_dis * d_loss + self.hp_cls * c_loss + self.hp_tra * t_loss
            all_losses[0] += cap_loss
            all_losses[1] += s_loss
            all_losses[2] += d_loss
            all_losses[3] += c_loss
            all_losses[4] += t_loss

            if training:
                if data_type == 'senti':
                    self.senti_optim.zero_grad()
                    s_loss.backward(retain_graph=True)
                    self.senti_optim.step()

                self.dis_optim.zero_grad()
                d_loss.backward(retain_graph=True)
                self.dis_optim.step()
                self.discriminator.weight_cliping()

                self.cls_optim.zero_grad()
                c_loss.backward(retain_graph=True)
                self.cls_optim.step()
                self.classifier.weight_cliping()

                self.tra_optim.zero_grad()
                t_loss.backward(retain_graph=True)
                self.tra_optim.step()
                self.translator.weight_cliping()

                self.cap_optim.zero_grad()
                cap_loss.backward()
                self.cap_optim.step()

            del fc_feats, att_feats, cpts_tensor, sentis_tensor, senti_labels

        return list((all_losses/len(data)).detach().numpy())

    def sample(self, fc_feats, att_feats, cpts_tensor, sentis_tensor,
               beam_size=3, decoding_constraint=1):
        self.eval()
        att_feats = att_feats.unsqueeze(1)
        _, senti_features, det_img_sentis, _ = self.senti_detector.sample(att_feats)
        captions, _ = self.captioner.sample(
            fc_feats, att_feats, cpts_tensor, senti_features, sentis_tensor,
            beam_size, decoding_constraint, self.max_seq_length)

        return captions, det_img_sentis
