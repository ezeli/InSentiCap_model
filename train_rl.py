# coding:utf8
import tqdm
import os
import time
import json
import sys
import pdb
import traceback
from bdb import BdbQuit
import torch
import numpy as np

from opts import parse_opt
from models.captioner import Captioner
from dataloader import get_caption_dataloader
from self_critical.utils import get_ciderd_scorer, get_self_critical_reward, RewardCriterion


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    idx2word = json.load(open(opt.idx2word, 'r'))
    img_captions = json.load(open(opt.img_captions, 'r'))
    img_det_concepts = json.load(open(opt.img_det_concepts, 'r'))

    captioner = Captioner(idx2word, opt.settings)
    captioner.to(opt.device)
    lr = 4e-5
    optimizer, _ = captioner.get_optim_criterion(lr)
    criterion = RewardCriterion()
    print("====> loading checkpoint '{}'".format(opt.rl_xe_resume))
    chkpoint = torch.load(opt.rl_xe_resume, map_location=lambda s, l: s)
    assert opt.settings == chkpoint['settings'], \
        'opt.settings and resume model settings are different'
    assert idx2word == chkpoint['idx2word'], \
        'idx2word and resume model idx2word are different'
    captioner.load_state_dict(chkpoint['model'])
    print("====> loaded checkpoint '{}', epoch: {}"
          .format(opt.rl_xe_resume, chkpoint['epoch']))

    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i

    print('====> process image captions begin')
    captions_id = {}
    for split, caps in img_captions.items():
        print('convert %s captions to index' % split)
        captions_id[split] = {}
        for fn, seqs in tqdm.tqdm(caps.items()):
            tmp = []
            for seq in seqs:
                tmp.append([captioner.sos_id] +
                           [word2idx.get(w, None) or word2idx['<UNK>'] for w in seq] +
                           [captioner.eos_id])
            captions_id[split][fn] = tmp
    img_captions = captions_id
    print('====> process image captions end')

    print('====> process image det_concepts begin')
    det_concepts_id = {}
    for fn, cpts in tqdm.tqdm(img_det_concepts.items()):
        det_concepts_id[fn] = [word2idx[w] for w in cpts]
    img_det_concepts = det_concepts_id
    print('====> process image det_concepts end')

    train_data = get_caption_dataloader(opt.fc_feats, opt.att_feats, img_captions['train'],
                                        img_det_concepts, idx2word.index('<PAD>'),
                                        opt.max_seq_len, opt.num_concepts,
                                        opt.xe_bs, opt.xe_num_works)
    val_data = get_caption_dataloader(opt.fc_feats, opt.att_feats, img_captions['val'],
                                      img_det_concepts, idx2word.index('<PAD>'),
                                      opt.max_seq_len, opt.num_concepts, opt.xe_bs,
                                      opt.xe_num_works, shuffle=False)
    test_captions = {}
    for fn in img_captions['test']:
        test_captions[fn] = [[]]
    test_data = get_caption_dataloader(opt.fc_feats, opt.att_feats, test_captions,
                                       img_det_concepts, idx2word.index('<PAD>'),
                                       opt.max_seq_len, opt.num_concepts, opt.xe_bs,
                                       opt.xe_num_works, shuffle=False)

    ciderd_scorer = get_ciderd_scorer(img_captions, captioner.sos_id, captioner.eos_id)

    checkpoint = os.path.join(opt.checkpoint, 'xe')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    result_dir = './result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    previous_loss = None
    for epoch in range(opt.xe_epochs):
        print('--------------------epoch: %d' % epoch)
        # torch.cuda.empty_cache()
        captioner.train()
        train_loss = 0.0
        avg_reward = 0.0
        for data_item in tqdm.tqdm(train_data):
            fns, fc_feats, att_feats, (caps_tensor, lengths), cpts_tensor = data_item
            fc_feats = fc_feats.to(opt.device)
            att_feats = att_feats.to(opt.device)
            cpts_tensor = cpts_tensor.to(opt.device)

            sample_captions, sample_logprobs, seq_masks = captioner(
                fc_feats, att_feats, cpts_tensor, None, None, sample_max=0, max_seq_len=opt.max_seq_len, mode='rl')

            captioner.eval()
            with torch.no_grad():
                greedy_captions, _, _ = captioner(
                fc_feats, att_feats, cpts_tensor, None, None, sample_max=1, max_seq_len=opt.max_seq_len, mode='rl')
            captioner.train()

            reward = get_self_critical_reward(
                sample_captions, greedy_captions, fns, img_captions['train'],
                captioner.sos_id, captioner.eos_id, ciderd_scorer)
            # reward = get_self_critical_reward(model, fc_feats, att_feats, None, data, sample_captions, None, new_CiderD_scorer=ciderd_scorer)
            loss = criterion(sample_logprobs, seq_masks, torch.from_numpy(reward).float().to(opt.device))
            avg_reward += np.mean(reward[:, 0]).item()

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_data)
        avg_reward /= len(train_data)

        # evaluation
        captioner.eval()
        with torch.no_grad():
            # test
            results = []
            for data_item in tqdm.tqdm(test_data):
                fns, fc_feats, att_feats, (caps_tensor, lengths), cpts_tensor = data_item
                fc_feats = fc_feats.to(opt.device)
                att_feats = att_feats.to(opt.device)
                cpts_tensor = cpts_tensor.to(opt.device)

                sents = captioner.sample(fc_feats, att_feats, cpts_tensor, beam_size=opt.beam_size, max_seq_len=opt.max_seq_len)[0]
                assert len(sents) == len(fns)
                for k, sent in enumerate(sents):
                    results.append({'image_id': fns[k], 'caption': sent})
            json.dump(results, open(os.path.join(result_dir, 'result_%d.json' % epoch), 'w'))

        print('train_loss:', train_loss, 'avg_reward:', avg_reward)

        if epoch in [0, 1, 2, 3, 5, 7, 9, 10, 12, 14, 15, 17, 19]:
            chkpoint = {
                'epoch': epoch,
                'model': captioner.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
            }
            checkpoint_path = os.path.join(
                checkpoint, 'model_%d_%.4f_%.4f_%s.pth' % (
                    epoch, train_loss, avg_reward, time.strftime('%m%d-%H%M')))
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
