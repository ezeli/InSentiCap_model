# coding:utf8
import tqdm
import os
import time
import json
from collections import defaultdict
import sys
import pdb
import traceback
from bdb import BdbQuit
import torch

from opts import parse_opt
from models.captioner import Captioner
from models.sent_senti_cls import SentenceSentimentClassifier
from dataloader import get_caption_dataloader, get_senti_corpus_with_sentis_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    dataset_name = opt.dataset_name
    corpus_type = opt.corpus_type

    idx2word = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'idx2word.json'), 'r'))
    img_captions = json.load(open(os.path.join(opt.captions_dir, dataset_name, 'img_captions.json'), 'r'))
    img_det_concepts = json.load(open(os.path.join(opt.captions_dir, dataset_name, 'img_det_concepts.json'), 'r'))
    senti_captions = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'senti_captions.json'), 'r'))

    captioner = Captioner(idx2word, opt.sentiment_categories, opt.settings)
    captioner.to(opt.device)
    lr = opt.xe_lr
    optimizer, xe_crit, da_crit = captioner.get_optim_criterion(lr)
    if opt.xe_resume:
        print("====> loading checkpoint '{}'".format(opt.xe_resume))
        chkpoint = torch.load(opt.xe_resume, map_location=lambda s, l: s)
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
        captioner.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.xe_resume, chkpoint['epoch']))

    sent_senti_cls = SentenceSentimentClassifier(idx2word, opt.sentiment_categories, opt.settings)
    sent_senti_cls.to(opt.device)
    ss_cls_file = os.path.join(opt.checkpoint, 'sent_senti_cls', dataset_name, corpus_type, 'model-best.pth')
    print("====> loading checkpoint '{}'".format(ss_cls_file))
    chkpoint = torch.load(ss_cls_file, map_location=lambda s, l: s)
    assert opt.settings == chkpoint['settings'], \
        'opt.settings and resume model settings are different'
    assert idx2word == chkpoint['idx2word'], \
        'idx2word and resume model idx2word are different'
    assert opt.sentiment_categories == chkpoint['sentiment_categories'], \
        'opt.sentiment_categories and resume model sentiment_categories are different'
    assert dataset_name == chkpoint['dataset_name'], \
        'dataset_name and resume model dataset_name are different'
    assert corpus_type == chkpoint['corpus_type'], \
        'corpus_type and resume model corpus_type are different'
    sent_senti_cls.load_state_dict(chkpoint['model'])
    sent_senti_cls.eval()

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

    senti_label2idx = {}
    for i, w in enumerate(opt.sentiment_categories):
        senti_label2idx[w] = i
    print('====> process senti corpus begin')
    senti_captions['positive'] = senti_captions['positive'] * int(len(senti_captions['neutral']) / len(senti_captions['positive']))
    senti_captions['negative'] = senti_captions['negative'] * int(len(senti_captions['neutral']) / len(senti_captions['negative']))
    senti_captions_id = []
    for senti, caps in senti_captions.items():
        print('convert %s corpus to index' % senti)
        senti_id = senti_label2idx[senti]
        for cap, cpts, sentis in tqdm.tqdm(caps):
            cap = [captioner.sos_id] +\
                  [word2idx.get(w, None) or word2idx['<UNK>'] for w in cap] +\
                  [captioner.eos_id]
            cpts = [word2idx[w] for w in cpts if w in word2idx]
            sentis = [word2idx[w] for w in sentis]
            senti_captions_id.append([cap, cpts, sentis, senti_id])
    senti_captions = senti_captions_id
    print('====> process senti corpus end')

    fc_feats = os.path.join(opt.feats_dir, dataset_name, '%s_fc.h5' % dataset_name)
    att_feats = os.path.join(opt.feats_dir, dataset_name, '%s_att.h5' % dataset_name)
    train_data = get_caption_dataloader(fc_feats, att_feats, img_captions['train'],
                                        img_det_concepts, idx2word.index('<PAD>'),
                                        opt.max_seq_len, opt.num_concepts,
                                        opt.xe_bs, opt.xe_num_works)
    val_data = get_caption_dataloader(fc_feats, att_feats, img_captions['val'],
                                      img_det_concepts, idx2word.index('<PAD>'),
                                      opt.max_seq_len, opt.num_concepts, opt.xe_bs,
                                      opt.xe_num_works, shuffle=False)
    scs_data = get_senti_corpus_with_sentis_dataloader(
        senti_captions, idx2word.index('<PAD>'), opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, 80, opt.xe_num_works)

    test_captions = {}
    for fn in img_captions['test']:
        test_captions[fn] = [[]]
    test_data = get_caption_dataloader(fc_feats, att_feats, test_captions,
                                       img_det_concepts, idx2word.index('<PAD>'),
                                       opt.max_seq_len, opt.num_concepts, opt.xe_bs,
                                       opt.xe_num_works, shuffle=False)

    def forward(data, training=True, ss_prob=0.0):
        captioner.train(training)
        if training:
            seq2seq_data = iter(scs_data)
        loss_val = defaultdict(float)
        for _, fc_feats, att_feats, (caps_tensor, lengths), cpts_tensor in tqdm.tqdm(data):
            fc_feats = fc_feats.to(opt.device)
            att_feats = att_feats.to(opt.device)
            caps_tensor = caps_tensor.to(opt.device)
            cpts_tensor = cpts_tensor.to(opt.device)

            with torch.no_grad():
                xe_senti_labels, _ = sent_senti_cls(caps_tensor[:, 1:], lengths)
                xe_senti_labels = xe_senti_labels.softmax(dim=-1)
                xe_senti_labels = xe_senti_labels.argmax(dim=-1).detach()

            pred = captioner(fc_feats, att_feats, cpts_tensor, caps_tensor,
                             xe_senti_labels, ss_prob, mode='xe')
            xe_loss = xe_crit(pred, caps_tensor[:, 1:], lengths)
            da_loss = da_crit(captioner.cpt_feats, captioner.fc_feats.detach())
            cap_loss = xe_loss + da_loss
            loss_val['xe_loss'] += float(xe_loss)
            loss_val['da_loss'] += float(da_loss)
            loss_val['cap_loss'] += float(cap_loss)

            seq2seq_loss = 0.0
            if training:
                try:
                    (caps_tensor, lengths), cpts_tensor, sentis_tensor, senti_labels = next(seq2seq_data)
                except:
                    seq2seq_data = iter(scs_data)
                    (caps_tensor, lengths), cpts_tensor, sentis_tensor, senti_labels = next(seq2seq_data)
                caps_tensor = caps_tensor.to(opt.device)
                cpts_tensor = cpts_tensor.to(opt.device)
                sentis_tensor = sentis_tensor.to(opt.device)
                senti_labels = senti_labels.to(opt.device)
                pred = captioner(caps_tensor, cpts_tensor, sentis_tensor, senti_labels,
                                 ss_prob, mode='seq2seq')
                seq2seq_loss = xe_crit(pred, caps_tensor[:, 1:], lengths)
                loss_val['seq2seq_loss'] += float(seq2seq_loss)

            all_loss = cap_loss + seq2seq_loss
            loss_val['all_loss'] += float(all_loss)

            if training:
                optimizer.zero_grad()
                all_loss.backward()
                clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()

        for k, v in loss_val.items():
            loss_val[k] = v / len(data)
        return loss_val

    tmp_dir = ''
    checkpoint = os.path.join(opt.checkpoint, 'xe', dataset_name, corpus_type, tmp_dir)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    result_dir = os.path.join(opt.result_dir, 'xe', dataset_name, corpus_type, tmp_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    previous_loss = None
    for epoch in range(opt.xe_epochs):
        print('--------------------epoch: %d' % epoch)
        # torch.cuda.empty_cache()
        ss_prob = 0.0
        if epoch > opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
        print('tmp_dir:', tmp_dir, 'ss_prob:', ss_prob)
        train_loss = forward(train_data, ss_prob=ss_prob)

        with torch.no_grad():
            val_loss = forward(val_data, training=False)

            results = []
            fact_txt = ''
            for fns, fc_feats, att_feats, _, _ in tqdm.tqdm(test_data):
                fc_feats = fc_feats.to(opt.device)
                att_feats = att_feats.to(opt.device)
                for i, fn in enumerate(fns):
                    captions, _ = captioner.sample(
                        fc_feats[i], att_feats[i],
                        beam_size=opt.beam_size, max_seq_len=opt.max_seq_len)
                    results.append({'image_id': fn, 'caption': captions[0]})
                    fact_txt += captions[0] + '\n'
            json.dump(results, open(os.path.join(result_dir, 'result_%d.json' % epoch), 'w'))
            with open(os.path.join(result_dir, 'result_%d.txt' % epoch), 'w') as f:
                f.write(fact_txt)

        if previous_loss is not None and val_loss['all_loss'] > previous_loss:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = val_loss['all_loss']

        print('train_loss: %s, val_loss: %s' % (dict(train_loss), dict(val_loss)))
        if epoch in [0, 10, 15, 20, 25, 29, 30, 35, 39]:
            chkpoint = {
                'epoch': epoch,
                'model': captioner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'sentiment_categories': opt.sentiment_categories,
                'dataset_name': dataset_name,
                'corpus_type': corpus_type,
            }
            checkpoint_path = os.path.join(checkpoint, 'model_%d_%.4f_%.4f_%s.pth' % (
                epoch, train_loss['all_loss'], val_loss['all_loss'], time.strftime('%m%d-%H%M')))
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
