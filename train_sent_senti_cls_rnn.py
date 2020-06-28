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

from opts import parse_opt
from models.sent_senti_cls import SentenceSentimentClassifier
from dataloader import get_senti_sents_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    senti_corpus = json.load(open(opt.real_captions, 'r'))['senti']
    idx2word = json.load(open(opt.idx2word, 'r'))
    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i
    senti_label2idx = {}
    for i, w in enumerate(opt.sentiment_categories):
        senti_label2idx[w] = i

    print('====> process senti_corpus begin')
    train_set = []
    val_set = []
    for senti in opt.sentiment_categories:
        print('convert %s corpus to index' % senti)
        senti_id = senti_label2idx[senti]
        sents = deepcopy(senti_corpus[senti])
        random.shuffle(sents)
        for sent in sents[1000:]:
            tmp = [word2idx.get(w, None) or word2idx['<UNK>'] for w in sent] + [word2idx['<EOS>']]
            train_set.append([senti_id, tmp])
        for sent in sents[:1000]:
            tmp = [word2idx.get(w, None) or word2idx['<UNK>'] for w in sent] + [word2idx['<EOS>']]
            val_set.append([senti_id, tmp])
    print('====> process senti_corpus end')

    train_data = get_senti_sents_dataloader(train_set, word2idx['<PAD>'], opt.max_seq_len)
    val_data = get_senti_sents_dataloader(val_set, word2idx['<PAD>'], opt.max_seq_len, shuffle=False)

    model = SentenceSentimentClassifier(idx2word, opt.sentiment_categories, opt.settings)
    model.to(opt.device)
    lr = 0.01
    optimizer, criterion = model.get_optim_and_crit(lr)

    checkpoint = os.path.join(opt.checkpoint, 'sent_senti_cls')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    previous_wrong_rate = None
    for epoch in range(30):
        print('--------------------epoch: %d' % epoch)
        model.train()
        train_loss = 0.0
        for sentis, (caps_tensor, lengths) in tqdm.tqdm(train_data):
            sentis = sentis.to(opt.device)
            caps_tensor = caps_tensor.to(opt.device)

            pred = model(caps_tensor, lengths)
            loss = criterion(pred, sentis)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, opt.grad_clip)
            optimizer.step()
        train_loss /= len(train_data)

        model.eval()
        all_num = 0
        wrong_num = 0
        with torch.no_grad():
            for sentis, (caps_tensor, lengths) in tqdm.tqdm(val_data):
                sentis = sentis.to(opt.device)
                caps_tensor = caps_tensor.to(opt.device)

                rest, _ = model.sample(caps_tensor, lengths)
                rest = torch.LongTensor(np.array(rest)).to(opt.device)
                all_num += int(sentis.size(0))
                wrong_num += int((sentis != rest).sum())
        wrong_rate = wrong_num / all_num

        if previous_wrong_rate is not None and wrong_rate > previous_wrong_rate:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_wrong_rate = wrong_rate

        print('train_loss: %.4f, wrong_rate: %.4f' % (train_loss, wrong_rate))
        if epoch in [0, 5, 10, 15, 20, 25, 29, 30, 35, 39]:
            chkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'sentiment_categories': opt.sentiment_categories,
            }
            checkpoint_path = os.path.join(checkpoint, 'model_%d_%.4f_%.4f_%s.pth' % (
                epoch, train_loss, wrong_rate, time.strftime('%m%d-%H%M')))
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
