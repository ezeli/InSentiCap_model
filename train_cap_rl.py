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

from opts import parse_opt
from models.captioner import Captioner
from dataloader import get_rl_fact_dataloader
from self_critical.utils import get_ciderd_scorer, get_self_critical_reward, RewardCriterion


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    idx2word = json.load(open(opt.idx2word, 'r'))
    img_captions = json.load(open(opt.img_captions, 'r'))
    img_det_concepts = json.load(open(opt.img_det_concepts, 'r'))
    img_det_sentiments = json.load(open(opt.img_det_sentiments, 'r'))

    model = Captioner(idx2word, opt.settings)
    model.to(opt.device)
    lr = opt.rl_lrs['cap_lr']
    optimizer, _ = model.get_optim_criterion(lr)
    if opt.rl_resume:
        print("====> loading checkpoint '{}'".format(opt.rl_resume))
        chkpoint = torch.load(opt.rl_resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        model.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.rl_resume, chkpoint['epoch']))
    elif opt.rl_xe_resume:
        print("====> loading checkpoint '{}'".format(opt.rl_xe_resume))
        chkpoint = torch.load(opt.rl_xe_resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        model.load_state_dict(chkpoint['model'])
    else:
        raise Exception('rl_resume or rl_xe_resume is required!')

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
                tmp.append([model.sos_id] +
                           [word2idx.get(w, None) or word2idx['<UNK>'] for w in seq] +
                           [model.eos_id])
            captions_id[split][fn] = tmp
    img_captions = captions_id
    print('====> process image captions end')

    print('====> process image det_concepts begin')
    det_concepts_id = {}
    for fn, cpts in tqdm.tqdm(img_det_concepts.items()):
        det_concepts_id[fn] = [word2idx[w] for w in cpts]
    img_det_concepts = det_concepts_id
    print('====> process image det_concepts end')

    print('====> process image det_sentiments begin')
    det_sentiments_id = {}
    for fn, sentis in tqdm.tqdm(img_det_sentiments.items()):
        det_sentiments_id[fn] = [word2idx[w] for w in sentis]
    img_det_sentiments = det_sentiments_id
    print('====> process image det_concepts end')

    fact_train_data = get_rl_fact_dataloader(
        opt.fc_feats, opt.att_feats, img_captions['train'], img_det_concepts,
        img_det_sentiments, model.pad_id, opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works)
    fact_val_data = get_rl_fact_dataloader(
        opt.fc_feats, opt.att_feats, img_captions['val'], img_det_concepts,
        img_det_sentiments, model.pad_id, opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works, shuffle=False)
    fact_test_data = get_rl_fact_dataloader(
        opt.fc_feats, opt.att_feats, img_captions['test'], img_det_concepts,
        img_det_sentiments, model.pad_id, opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works, shuffle=False)

    ciderd_scorer = get_ciderd_scorer(img_captions, model.sos_id, model.eos_id)
    crit = RewardCriterion()

    checkpoint = os.path.join(opt.checkpoint, 'cap_rl')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    result_dir = './result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    previous_loss = None
    for epoch in range(opt.rl_epochs):
        print('--------------------epoch: %d' % epoch)
        torch.cuda.empty_cache()
        model.train()
        train_loss = 0.0
        avg_reward = 0.0
        for fns, fc_feats, att_feats, cpts_tensor, _, ground_truth in tqdm.tqdm(fact_train_data):
            fc_feats = fc_feats.to(opt.device)
            att_feats = att_feats.to(opt.device)
            cpts_tensor = cpts_tensor.to(opt.device)

            sample_captions, sample_logprobs, seq_masks = model(
                fc_feats, att_feats, cpts_tensor, opt.max_seq_len,
                sample_max=0, mode='cap_rl')
            model.eval()
            with torch.no_grad():
                greedy_captions, _, _ = model(
                    fc_feats, att_feats, cpts_tensor, opt.max_seq_len,
                    sample_max=1, mode='cap_rl')
            model.train()

            fact_reward = get_self_critical_reward(
                sample_captions, greedy_captions, fns, ground_truth,
                model.sos_id, model.eos_id, ciderd_scorer)
            fact_reward = torch.from_numpy(fact_reward).float().to(opt.device)
            loss = crit(sample_logprobs, seq_masks, fact_reward)

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()

            train_loss += loss.item()
            avg_reward += fact_reward[:, 0].sum().item()
        train_loss = train_loss / len(fact_train_data)
        avg_reward = avg_reward / len(fact_train_data)
        print('train loss:', train_loss, 'avg reward:', avg_reward)

        with torch.no_grad():
            torch.cuda.empty_cache()
            # test
            results = []
            for fns, fc_feats, att_feats, cpts_tensor, _, _ in \
                    tqdm.tqdm(fact_test_data):
                fc_feats = fc_feats.to(opt.device)
                att_feats = att_feats.to(opt.device)
                cpts_tensor = cpts_tensor.to(opt.device)

                for i, fn in enumerate(fns):
                    captions, _ = model.sample(
                        fc_feats[i], att_feats[i], cpts_tensor[i],
                        beam_size=opt.beam_size, max_seq_length=opt.max_seq_len)
                    results.append({'image_id': fn, 'caption': captions[0]})

            json.dump(results, open(os.path.join(result_dir, 'result_%d.json' % epoch), 'w'))

        # if previous_loss is not None and senti_val_loss[0] > previous_loss[0] \
        #         and fact_val_loss[0] > previous_loss[1]:
        #     for optim in [model.cap_optim, model.senti_optim]:
        #         lr = optim.param_groups[0]['lr'] * 0.5
        #         for param_group in optim.param_groups:
        #             param_group['lr'] = lr
        # previous_loss = [senti_val_loss[0], fact_val_loss[0]]

        if epoch in [0, 1, 2, 3, 5, 7, 9, 10, 12, 14, 15, 17, 19]:
            chkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
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
