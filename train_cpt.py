# coding:utf8
import tqdm
import os
from copy import deepcopy
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
    dataset_name = opt.dataset_name

    idx2concept = json.load(open(os.path.join(opt.captions_dir, dataset_name, 'idx2concept.json'), 'r'))
    img_concepts = json.load(open(os.path.join(opt.captions_dir, dataset_name, 'img_concepts.json'), 'r'))

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
        assert dataset_name == chkpoint['dataset_name'], \
            'dataset_name and resume model dataset_name are different'
        cpt_detector.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.concept_resume, chkpoint['epoch']))

    concept2idx = {}
    for i, w in enumerate(idx2concept):
        concept2idx[w] = i

    ground_truth = deepcopy(img_concepts['test'])
    print('====> process image concepts begin')
    img_concepts_id = {}
    for split, concepts in img_concepts.items():
        print('convert %s concepts to index' % split)
        img_concepts_id[split] = {}
        for fn, cpts in tqdm.tqdm(concepts.items()):
            cpts = [concept2idx[c] for c in cpts if c in concept2idx]
            img_concepts_id[split][fn] = cpts
    img_concepts = img_concepts_id
    print('====> process image concepts end')

    f_fc = os.path.join(opt.feats_dir, dataset_name, '%s_fc.h5' % dataset_name)
    train_data = get_concept_dataloader(
        f_fc, img_concepts['train'], len(idx2concept),
        opt.concept_bs, opt.concept_num_works)
    val_data = get_concept_dataloader(
        f_fc, img_concepts['val'], len(idx2concept),
        opt.concept_bs, opt.concept_num_works, shuffle=False)
    test_data = get_concept_dataloader(
        f_fc, img_concepts['test'], len(idx2concept),
        opt.concept_bs, opt.concept_num_works, shuffle=False)

    def forward(data, training=True):
        cpt_detector.train(training)
        loss_val = 0.0
        for _, fc_feats, cpts_tensors in tqdm.tqdm(data):
            fc_feats = fc_feats.to(opt.device)
            cpts_tensors = cpts_tensors.to(opt.device)
            pred = cpt_detector(fc_feats)
            loss = criterion(pred, cpts_tensors)
            loss_val += loss.item()
            if training:
                optimizer.zero_grad()
                loss.backward()
                clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
        return loss_val / len(data)

    checkpoint = os.path.join(opt.checkpoint, 'concept', dataset_name)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    previous_loss = None
    for epoch in range(opt.concept_epochs):
        print('--------------------epoch: %d' % epoch)
        train_loss = forward(train_data)
        with torch.no_grad():
            val_loss = forward(val_data, training=False)

            # test
            test_loss = 0.0
            pre = 0.0
            recall = 0.0
            last_score = 0.0
            for fns, fc_feats, cpts_tensors in tqdm.tqdm(test_data):
                fc_feats = fc_feats.to(opt.device)
                cpts_tensors = cpts_tensors.to(opt.device)
                pred, concepts, scores = cpt_detector.sample(fc_feats, num=opt.num_concepts)
                loss = criterion(pred, cpts_tensors)
                test_loss += loss.item()
                tmp_pre = 0.0
                tmp_rec = 0.0
                for i, fn in enumerate(fns):
                    cpts = concepts[i]
                    grdt = ground_truth[fn]
                    jiaoji = len(set(grdt) - (set(grdt) - set(cpts)))
                    tmp_pre += jiaoji / len(cpts)
                    tmp_rec += jiaoji / len(grdt)
                pre += tmp_pre / len(fns)
                recall += tmp_rec / len(fns)
                last_score += float(scores[:, -1].mean())
            data_len = len(test_data)
            test_loss = test_loss / data_len
            pre = pre / data_len
            recall = recall / data_len
            last_score = last_score / data_len

        if previous_loss is not None and val_loss > previous_loss:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = val_loss

        print('train_loss: %.4f, val_loss: %.4f, test_loss: %.4f, '
              'precision: %.4f, recall: %.4f, last_score: %.4f' %
              (train_loss, val_loss, test_loss, pre, recall, last_score))
        if epoch in [0, 1, 2, 5, 7, 9] or epoch > 10:
            chkpoint = {
                'epoch': epoch,
                'model': cpt_detector.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2concept': idx2concept,
                'dataset_name': dataset_name,
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
