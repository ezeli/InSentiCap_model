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
from models.concept_detector import ConceptDetector
from dataloader import get_concept_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    idx2concept = json.load(open(opt.idx2concept, 'r'))
    f_fc = h5py.File(opt.fc_feats, mode='r')
    img_concepts = json.load(open(opt.img_concepts, 'r'))

    cpt_detector = ConceptDetector(idx2concept, opt.settings)
    cpt_detector.to(opt.device)

    lr = opt.concept_lr
    optimizer, criterion = cpt_detector.get_optim_criterion(lr)
    if opt.concept_resume:
        print("====> loading checkpoint '{}'".format(opt.concept_resume))
        chkpoint = torch.load(opt.concept_resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2concept == chkpoint['idx2concept'], \
            'idx2concept and resume model idx2concept are different'
        cpt_detector.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.concept_resume, chkpoint['epoch']))

    train_data = get_concept_dataloader(f_fc, img_concepts['train'], idx2concept, opt.concept_bs)
    val_data = get_concept_dataloader(f_fc, img_concepts['val'], idx2concept, opt.concept_bs, shuffle=False)

    def forward(data, training=True):
        cpt_detector.train(training)
        loss_val = 0.0
        loss_list = []
        for _, fc_feats, cpts_tensors in tqdm.tqdm(data):
            fc_feats = fc_feats.to(opt.device)
            cpts_tensors = cpts_tensors.to(opt.device)
            pred = cpt_detector(fc_feats)
            loss = criterion(pred, cpts_tensors)
            loss_val += loss.item()
            loss_list.append(loss.item())
            if training:
                optimizer.zero_grad()
                loss.backward()
                # TODO
                clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
                # cpt_detector.weight_cliping(opt.grad_clip)
        return loss_val / len(data)

    checkpoint = os.path.join(opt.checkpoint, 'concept')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    previous_loss = None
    for epoch in range(opt.concept_epochs):
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
                'model': cpt_detector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2concept': idx2concept,
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
