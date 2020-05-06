# coding:utf8
import tqdm
import os
import h5py
import time
import json
import sys
import pdb
import traceback
from bdb import BdbQuit
import torch
from torch.nn.utils.rnn import pack_padded_sequence

from opts import parse_opt
from models.insenti_cap import InSentiCap
from dataloader import get_iter_fact_dataloader, get_iter_senti_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    idx2word = json.load(open(opt.idx2word, 'r'))
    img_captions = json.load(open(opt.img_captions, 'r'))
    img_det_concepts = json.load(open(opt.img_det_concepts, 'r'))
    img_det_sentiments = json.load(open(opt.img_det_sentiments, 'r'))
    img_senti_labels = json.load(open(opt.img_senti_labels, 'r'))
    real_captions = json.load(open(opt.real_captions, 'r'))
    f_fc = h5py.File(opt.fc_feats, mode='r')
    f_att = h5py.File(opt.att_feats, mode='r')
    f_senti_fc = h5py.File(opt.senti_fc_feats, mode='r')
    f_senti_att = h5py.File(opt.senti_att_feats, mode='r')

    fact_train_data = get_iter_fact_dataloader(
        f_fc, f_att, img_det_concepts, img_det_sentiments, idx2word,
        img_captions['train'].keys(), idx2word.index('<PAD>'),
        opt.num_concepts, opt.num_sentiments, opt.iter_bs)
    fact_val_data = get_iter_fact_dataloader(
        f_fc, f_att, img_det_concepts, img_det_sentiments, idx2word,
        img_captions['val'].keys(), idx2word.index('<PAD>'),
        opt.num_concepts, opt.num_sentiments, opt.iter_bs, shuffle=False)
    del img_captions

    senti_train_data = get_iter_senti_dataloader(
        f_senti_fc, f_senti_att, img_det_concepts, img_det_sentiments, idx2word,
        img_senti_labels['train'], opt.sentiment_categories, idx2word.index('<PAD>'),
        opt.num_concepts, opt.num_sentiments, opt.iter_bs)
    senti_val_data = get_iter_senti_dataloader(
        f_senti_fc, f_senti_att, img_det_concepts, img_det_sentiments, idx2word,
        img_senti_labels['val'], opt.sentiment_categories, idx2word.index('<PAD>'),
        opt.num_concepts, opt.num_sentiments, opt.iter_bs, shuffle=False)

    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i

    real_captions_tmp = {'fact': [], 'senti': []}
    for cap in real_captions['fact']:
        real_captions_tmp['fact'].append(
            [word2idx.get(w, None) or word2idx['<UNK>'] for w in cap])
    for senti_label, caps in real_captions['senti'].items():
        for cap in caps:
            real_captions_tmp['senti'].append(
                (opt.sentiment_categories.index(senti_label),
                 [word2idx.get(w, None) or word2idx['<UNK>'] for w in cap]))
    real_captions = real_captions_tmp

    in_senti_cap = InSentiCap(
        idx2word, opt.max_sql_len, opt.sentiment_categories, opt.iter_lrs,
        opt.iter_hyperparams, real_captions, opt.settings)
    in_senti_cap.to(opt.device)

    if opt.iter_resume:
        print("====> loading checkpoint '{}'".format(opt.iter_resume))
        chkpoint = torch.load(opt.iter_resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        assert opt.max_seq_length == chkpoint['max_seq_length'], \
            'opt.max_seq_length and resume model max_seq_length are different'
        assert opt.sentiment_categories == chkpoint['sentiment_categories'], \
            'opt.sentiment_categories and resume model sentiment_categories are different'
        assert opt.iter_hyperparams == chkpoint['iter_hyperparams'], \
            'opt.iter_hyperparams and resume model iter_hyperparams are different'
        in_senti_cap.load_state_dict(chkpoint['model'])
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.iter_resume, chkpoint['epoch']))
    elif opt.iter_xe_resume:
        print("====> loading checkpoint '{}'".format(opt.iter_xe_resume))
        chkpoint = torch.load(opt.iter_xe_resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        in_senti_cap.captioner.load_state_dict(chkpoint['model'])
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.iter_xe_resume, chkpoint['epoch']))
    else:
        raise Exception('iter_resume or iter_xe_resume is required!')

    checkpoint = os.path.join(opt.checkpoint, 'iter')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    previous_loss = None
    for epoch in range(opt.iter_epochs):
        print('--------------------epoch: %d' % epoch)
        # torch.cuda.empty_cache()
        for i in range(opt.iter_senti_times):
            print('----------iter_senti_times: %d' % i)
            senti_train_loss = in_senti_cap(senti_train_data, data_type='senti', training=True)
            print('senti_train_loss: %s' % senti_train_loss)
        for i in range(opt.iter_fact_times):
            print('----------iter_fact_times: %d' % i)
            fact_train_loss = in_senti_cap(fact_train_data, data_type='fact', training=True)
            print('fact_train_loss: %s' % fact_train_loss)
        with torch.no_grad():
            print('----------val')
            senti_val_loss = in_senti_cap(senti_val_data, data_type='senti', training=False)
            print('senti_val_loss: %s' % senti_val_loss)
            fact_val_loss = in_senti_cap(fact_val_data, data_type='fact', training=False)
            print('fact_val_loss: %s' % fact_val_loss)

        if previous_loss is not None and senti_val_loss[0] > previous_loss[0] \
                and fact_val_loss[0] > previous_loss[1]:
            for optim in [in_senti_cap.cap_optim, in_senti_cap.senti_optim,
                          in_senti_cap.dis_optim, in_senti_cap.cls_optim,
                          in_senti_cap.tra_optim]:
                lr = optim.param_groups[0]['lr'] * 0.5
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
        previous_loss = [senti_val_loss[0], fact_val_loss[0]]

        print('senti_train_loss: %.4f, fact_train_loss: %.4f, '
              'senti_val_loss: %.4f, fact_val_loss: %.4f' %
              (senti_train_loss[0], fact_train_loss[0],
               senti_val_loss[0], fact_val_loss[0]))
        if epoch > -1:
            chkpoint = {
                'epoch': epoch,
                'model': in_senti_cap.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'max_seq_length': opt.max_seq_length,
                'sentiment_categories': opt.sentiment_categories,
                'iter_hyperparams': opt.iter_hyperparams,
            }
            checkpoint_path = os.path.join(
                checkpoint, 'model_%d_%.4f_%.4f_%.4f_%.4f_%s.pth' % (
                epoch, senti_train_loss[0], fact_train_loss[0], senti_val_loss[0],
                fact_val_loss[0], time.strftime('%m%d-%H%M')))
            torch.save(chkpoint, checkpoint_path)


if __name__ == '__main__':
    try:
        opt = parse_opt()
        train()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
