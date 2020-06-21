import torch
import torch.nn as nn
import tqdm

from .captioner import Captioner
from .sentiment_detector import SentimentDetector
# import sys
# sys.path.append("../")
from self_critical.utils import get_ciderd_scorer, get_self_critical_reward, \
    get_lm_reward, RewardCriterion


def clip_gradient(optimizer, grad_clip=0.1):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


class Detector(nn.Module):
    def __init__(self, idx2word, max_seq_len, sentiment_categories, lrs, settings):
        super(Detector, self).__init__()
        self.idx2word = idx2word
        self.pad_id = idx2word.index('<PAD>')
        self.max_seq_len = max_seq_len

        self.captioner = Captioner(idx2word, settings)
        self.senti_detector = SentimentDetector(sentiment_categories, settings)

        self.cap_optim, self.cap_xe_crit = self.captioner.get_optim_criterion(lrs['cap_lr'])
        self.cap_rl_crit = RewardCriterion()
        self.senti_optim, self.senti_crit = self.senti_detector.get_optim_criterion(lrs['senti_lr'])

        self.ciderd_scorer = None
        self.lms = {}

    def set_ciderd_scorer(self, captions):
        self.ciderd_scorer = get_ciderd_scorer(captions, self.captioner.sos_id, self.captioner.eos_id)

    def set_lms(self, lms):
        self.lms = lms

    def forward(self, data, data_type, training):
        self.train(training)
        all_losses = [0.0, 0.0, 0.0]
        device = next(self.parameters()).device
        for data_item in tqdm.tqdm(data):
            if data_type == 'fact':
                fns, fc_feats, att_feats, (caps_tensor, lengths), cpts_tensor, sentis_tensor, ground_truth = data_item
            elif data_type == 'senti':
                fns, fc_feats, att_feats, cpts_tensor, sentis_tensor, senti_labels = data_item
                senti_labels = senti_labels.to(device)
            else:
                raise Exception('data_type(%s) is wrong!' % data_type)

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

            sample_captions, sample_logprobs, seq_masks = self.captioner(
                fc_feats, att_feats, cpts_tensor, det_senti_features,
                sentis_tensor, self.max_seq_len, sample_max=0, mode='rl')
            self.eval()
            with torch.no_grad():
                greedy_captions, _, _ = self.captioner(
                    fc_feats, att_feats, cpts_tensor, det_senti_features,
                    sentis_tensor, self.max_seq_len, sample_max=1, mode='rl')
            self.train(training)

            if data_type == 'fact':
                fact_reward = get_self_critical_reward(
                    sample_captions, greedy_captions, fns, ground_truth,
                    self.captioner.sos_id, self.captioner.eos_id, self.ciderd_scorer)
                fact_reward = torch.from_numpy(fact_reward).float().to(device)
            else:
                fact_reward = 0

            senti_reward = get_lm_reward(
                sample_captions, greedy_captions, senti_labels,
                self.captioner.sos_id, self.captioner.eos_id, self.lms)
            senti_reward = torch.from_numpy(senti_reward).float().to(device)

            rewards = senti_reward + fact_reward
            # rewards = fact_reward
            cap_loss = self.cap_rl_crit(sample_logprobs, seq_masks, rewards)
            
            all_losses[0] += float(senti_reward[:, 0].sum())
            if data_type == 'fact':
                all_losses[1] += float(fact_reward[:, 0].sum())
            all_losses[2] += float(s_loss)

            if training:
                if data_type == 'senti':
                    self.senti_optim.zero_grad()
                    s_loss.backward(retain_graph=True)
                    clip_gradient(self.senti_optim)
                    self.senti_optim.step()

                self.cap_optim.zero_grad()
                cap_loss.backward()
                clip_gradient(self.cap_optim)
                self.cap_optim.step()

        return - all_losses[0] / len(data), - all_losses[1] / len(data), all_losses[2] / len(data)

    def sample(self, fc_feats, att_feats, cpts_tensor, sentis_tensor,
               beam_size=3, decoding_constraint=1):
        self.eval()
        att_feats = att_feats.unsqueeze(0)
        _, senti_features, det_img_sentis, _ = self.senti_detector.sample(att_feats)
        captions, _ = self.captioner.sample(
            fc_feats, att_feats, cpts_tensor, senti_features, sentis_tensor,
            beam_size, decoding_constraint, self.max_seq_len)

        return captions, det_img_sentis
