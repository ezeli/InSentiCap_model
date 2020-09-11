import torch
import numpy as np
import os
import time
import sys
import pdb
import traceback
from bdb import BdbQuit
import json
import random
import tqdm
from copy import deepcopy
from collections import defaultdict

from opts import parse_opt
from models.sent_senti_cls import SentenceSentimentClassifier
from dataloader import get_senti_sents_dataloader


random.seed(100)
resume = ''


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    dataset_name = opt.dataset_name
    corpus_type = opt.corpus_type

    idx2word = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'idx2word.json'), 'r'))
    senti_captions = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'senti_captions.json'), 'r'))

    model = SentenceSentimentClassifier(idx2word, opt.sentiment_categories, opt.settings)
    model.to(opt.device)
    lr = 4e-4
    optimizer, criterion = model.get_optim_and_crit(lr)
    if resume:
        print("====> loading checkpoint '{}'".format(resume))
        chkpoint = torch.load(resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        assert opt.sentiment_categories == chkpoint['sentiment_categories'], \
            'sentiment_categories and resume model sentiment_categories are different'
        assert dataset_name == chkpoint['dataset_name'], \
            'dataset_name and resume model dataset_name are different'
        assert corpus_type == chkpoint['corpus_type'], \
            'corpus_type and resume model corpus_type are different'
        model.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(resume, chkpoint['epoch']))

    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i
    senti_label2idx = {}
    for i, w in enumerate(opt.sentiment_categories):
        senti_label2idx[w] = i

    print('====> process senti_corpus begin')
    for senti in senti_captions:
        senti_captions[senti] = [c[0] for c in senti_captions[senti]]
        random.shuffle(senti_captions[senti])
    tmp_senti_captions = {'train': {}, 'val': {}}
    tmp_senti_captions['train']['neutral'] = deepcopy(senti_captions['neutral'][5000:])
    tmp_senti_captions['val']['neutral'] = deepcopy(senti_captions['neutral'][:5000])
    tmp_senti_captions['train']['positive'] = deepcopy(senti_captions['positive'][1000:])
    tmp_senti_captions['val']['positive'] = deepcopy(senti_captions['positive'][:1000])
    tmp_senti_captions['train']['negative'] = deepcopy(senti_captions['negative'][1000:])
    tmp_senti_captions['val']['negative'] = deepcopy(senti_captions['negative'][:1000])
    tmp_senti_captions['train']['positive'] = tmp_senti_captions['train']['positive'] * int(len(tmp_senti_captions['train']['neutral']) / len(tmp_senti_captions['train']['positive']))
    tmp_senti_captions['train']['negative'] = tmp_senti_captions['train']['negative'] * int(len(tmp_senti_captions['train']['neutral']) / len(tmp_senti_captions['train']['negative']))
    senti_captions = tmp_senti_captions

    train_set = []
    val_set = {}
    for senti in opt.sentiment_categories:
        print('convert %s corpus to index' % senti)
        senti_id = senti_label2idx[senti]
        for cap in tqdm.tqdm(senti_captions['train'][senti]):
            tmp = [word2idx.get(w, None) or word2idx['<UNK>'] for w in cap] + [word2idx['<EOS>']]
            train_set.append([senti_id, tmp])
        val_set[senti] = []
        for cap in tqdm.tqdm(senti_captions['val'][senti]):
            tmp = [word2idx.get(w, None) or word2idx['<UNK>'] for w in cap] + [word2idx['<EOS>']]
            val_set[senti].append([senti_id, tmp])
    random.shuffle(train_set)
    print('====> process senti_corpus end')

    train_data = get_senti_sents_dataloader(train_set, word2idx['<PAD>'], opt.max_seq_len)
    val_data = {}
    for senti in val_set:
        val_data[senti] = get_senti_sents_dataloader(val_set[senti], word2idx['<PAD>'], opt.max_seq_len, shuffle=False)

    checkpoint = os.path.join(opt.checkpoint, 'sent_senti_cls', dataset_name, corpus_type)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    result_dir = os.path.join(opt.result_dir, 'sent_senti_cls', dataset_name, corpus_type)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    previous_acc_rate = None
    for epoch in range(30):
        print('--------------------epoch: %d' % epoch)
        model.train()
        train_loss = 0.0
        for sentis, (caps_tensor, lengths) in tqdm.tqdm(train_data):
            sentis = sentis.to(opt.device)
            caps_tensor = caps_tensor.to(opt.device)

            pred, _ = model(caps_tensor, lengths)
            loss = criterion(pred, sentis)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
        train_loss /= len(train_data)

        model.eval()
        all_num = defaultdict(int)
        senti_num = {}
        test_case = defaultdict(list)
        for senti in opt.sentiment_categories:
            senti_num[senti] = defaultdict(int)
        with torch.no_grad():
            for senti, data in val_data.items():
                for sentis, (caps_tensor, lengths) in tqdm.tqdm(data):
                    sentis = sentis.to(opt.device)
                    caps_tensor = caps_tensor.to(opt.device)

                    rest, rest_w, att_weights = model.sample(caps_tensor, lengths)
                    rest = torch.LongTensor(np.array(rest)).to(opt.device)
                    total_num = int(sentis.size(0))
                    wrong_num = int((sentis != rest).sum())
                    all_num['total_num'] += total_num
                    all_num['wrong_num'] += wrong_num
                    senti_num[senti]['total_num'] += total_num
                    senti_num[senti]['wrong_num'] += wrong_num

                    random_id = random.randint(0, caps_tensor.size(0)-1)
                    caption = ' '.join([idx2word[idx] for idx in caps_tensor[random_id]])
                    pred_senti = rest_w[random_id]
                    att_weight = str(att_weights[random_id].detach().cpu().numpy().tolist())
                    test_case[senti].append([caption, pred_senti, att_weight])

        tmp_total_num = 0
        tmp_wrong_num = 0
        for senti in senti_num:
            tmp_total_num += senti_num[senti]['total_num']
            tmp_wrong_num += senti_num[senti]['wrong_num']
        assert tmp_total_num == all_num['total_num'] and tmp_wrong_num == all_num['wrong_num']

        all_acc_rate = 100 - all_num['wrong_num'] / all_num['total_num'] * 100
        senti_acc_rate = {}
        for senti in senti_num:
            senti_acc_rate[senti] = 100 - senti_num[senti]['wrong_num'] / senti_num[senti]['total_num'] * 100

        json.dump(test_case, open(os.path.join(result_dir, 'test_case_%d_%.4f.json' % (epoch, all_acc_rate)), 'w'))

        if previous_acc_rate is not None and all_acc_rate < previous_acc_rate:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_acc_rate = all_acc_rate

        print('train_loss: %.4f, all_acc_rate: %.4f, senti_acc_rate: %s' %
              (train_loss, all_acc_rate, senti_acc_rate))
        if epoch > -1:
            chkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'sentiment_categories': opt.sentiment_categories,
                'dataset_name': dataset_name,
                'corpus_type': corpus_type,
            }
            checkpoint_path = os.path.join(checkpoint, 'model_%d_%.4f_%.4f_%s.pth' % (
                epoch, train_loss, all_acc_rate, time.strftime('%m%d-%H%M')))
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
