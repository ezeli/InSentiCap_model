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

from opts import parse_opt
from models.sentiment_detector import SentimentDetector
from dataloader import get_senti_image_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    f_senti_att = h5py.File(opt.senti_att_feats, mode='r')
    img_senti_labels = json.load(open(opt.img_senti_labels, 'r'))

    senti_detector = SentimentDetector(opt.sentiment_categories, opt.settings)
    senti_detector.to(opt.device)

    lr = opt.senti_lr
    optimizer, criterion = senti_detector.get_optim_criterion(lr)
    if opt.senti_resume:
        print("====> loading checkpoint '{}'".format(opt.senti_resume))
        chkpoint = torch.load(opt.senti_resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert opt.sentiment_categories == chkpoint['sentiment_categories'], \
            'sentiment_categories and resume model sentiment_categories are different'
        senti_detector.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.senti_resume, chkpoint['epoch']))

    train_data = get_senti_image_dataloader(f_senti_att, img_senti_labels['train'],
                                            opt.sentiment_categories, opt.senti_bs)
    val_data = get_senti_image_dataloader(f_senti_att, img_senti_labels['val'],
                                          opt.sentiment_categories, opt.senti_bs, shuffle=False)

    def forward(data, training=True):
        senti_detector.train(training)
        loss_val = 0.0
        for _, att_feats, labels in tqdm.tqdm(data):
            att_feats = att_feats.to(opt.device)
            labels = labels.to(opt.device)
            pred, _ = senti_detector(att_feats)
            loss = criterion(pred, labels)
            loss_val += loss.item()
            if training:
                optimizer.zero_grad()
                loss.backward()
                # TODO
                # clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
        return loss_val / len(data)

    checkpoint = os.path.join(opt.checkpoint, 'sentiment')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    previous_loss = None
    for epoch in range(opt.senti_epochs):
        print('--------------------epoch: %d' % epoch)
        # torch.cuda.empty_cache()
        train_loss = forward(train_data)
        with torch.no_grad():
            val_loss = forward(val_data, training=False)

        if previous_loss is not None and val_loss > previous_loss:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = val_loss

        print('train_loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
        if epoch in [0, 1, 2, 5, 7, 9] or epoch > 10:
            chkpoint = {
                'epoch': epoch,
                'model': senti_detector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'sentiment_categories': opt.sentiment_categories,
            }
            checkpoint_path = os.path.join(checkpoint, 'model_%d_%.4f_%.4f_%s.pth' % (
                epoch, train_loss, val_loss, time.strftime('%m%d-%H%M')))
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
