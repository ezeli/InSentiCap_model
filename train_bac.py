# coding:utf8
import tqdm
import os
import h5py
import time
import json
import numpy as np
import sys
import pdb
import traceback
from bdb import BdbQuit
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from dataloader import get_dataloader
from models.captioner import Captioner
from opts import parse_opt
from self_critical.utils import get_ciderd_scorer, get_self_critical_reward, RewardCriterion


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    opt = parse_opt()
    train_mode = opt.train_mode

    idx2word = json.load(open(opt.idx2word, 'r'))
    captions = json.load(open(opt.captions, 'r'))
    f_fc = h5py.File(opt.img_feats, mode='r')

    # 模型
    decoder = Captioner(idx2word, opt.settings)
    decoder.to(opt.device)
    lr = opt.learning_rate
    optimizer = decoder.get_optimizer(lr)
    if opt.resume:
        print("====> loading checkpoint '{}'".format(opt.resume))
        chkpoint = torch.load(opt.resume, map_location=lambda s, l: s)
        decoder.load_state_dict(chkpoint['model'])
        if chkpoint['train_mode'] == train_mode:
            optimizer.load_state_dict(chkpoint['optimizer'])
            lr = optimizer.param_groups[0]['lr']
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        print("====> loaded checkpoint '{}', epoch: {}, train_mode: {}"
              .format(opt.resume, chkpoint['epoch'], chkpoint['train_mode']))
    elif train_mode == 'rl':
        raise Exception('"rl" mode need resume model')

    train_data = get_dataloader(f_fc, captions['train'], decoder.pad_id, opt.max_sql_len+1, opt.batch_size)
    val_data = get_dataloader(f_fc, captions['val'], decoder.pad_id, opt.max_sql_len+1, opt.batch_size, shuffle=False)

    xe_criterion = nn.CrossEntropyLoss()
    if train_mode == 'rl':
        rl_criterion = RewardCriterion()
        ciderd_scorer = get_ciderd_scorer(captions, decoder.sos_id, decoder.eos_id)

    def forward(data, training=True, ss_prob=0.0):
        decoder.train(training)
        loss_val = 0.0
        reward_val = 0.0
        for fns, fc_feats, (caps_tensor, lengths) in tqdm.tqdm(data):
            fc_feats = fc_feats.to(opt.device)
            caps_tensor = caps_tensor.to(opt.device)

            if training and train_mode == 'rl':
                sample_captions, sample_logprobs = decoder(fc_feats, sample_max=0,
                                                           max_seq_len=opt.max_sql_len, mode=train_mode)
                decoder.eval()
                with torch.no_grad():
                    greedy_captions, _ = decoder(fc_feats, sample_max=1,
                                                 max_seq_len=opt.max_sql_len, mode=train_mode)
                decoder.train()
                reward = get_self_critical_reward(sample_captions, greedy_captions, fns, captions['train'],
                                                  decoder.sos_id, decoder.eos_id, ciderd_scorer)
                loss = rl_criterion(sample_captions, sample_logprobs, torch.from_numpy(reward).float().to(opt.device))
                reward_val += np.mean(reward[:, 0]).item()
            else:
                pred = decoder(fc_feats, caps_tensor, lengths, ss_prob=ss_prob)
                real = pack_padded_sequence(caps_tensor[:, 1:], lengths, batch_first=True)[0]
                loss = xe_criterion(pred, real)

            loss_val += loss.item()
            if training:
                optimizer.zero_grad()
                loss.backward()
                clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
        return loss_val / len(data), reward_val / len(data)

    previous_loss = None
    for epoch in range(opt.max_epochs):
        print('--------------------epoch: %d' % epoch)
        # torch.cuda.empty_cache()
        ss_prob = 0.0
        if epoch > opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
        train_loss, avg_reward = forward(train_data, ss_prob=ss_prob)
        with torch.no_grad():
            val_loss, _ = forward(val_data, training=False)

        if train_mode != 'rl':
            if previous_loss is not None and val_loss >= previous_loss:
                lr = lr * 0.5
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            previous_loss = val_loss

        print('train_loss: %.4f, val_loss: %.4f, avg_reward: %.4f' % (train_loss, val_loss, avg_reward))
        if epoch in [5, 15, 18, 20, 23, 25, 27, 30, 33, 36, 39]:
            chkpoint = {
                'epoch': epoch,
                'model': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'train_mode': train_mode,
            }
            checkpoint_path = os.path.join(opt.checkpoint, 'model_%d_%.4f_%.4f_%.4f_%s.pth' % (
                epoch, train_loss, val_loss, avg_reward, time.strftime('%m%d-%H%M')))
            torch.save(chkpoint, checkpoint_path)


if __name__ == '__main__':
    try:
        train()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
