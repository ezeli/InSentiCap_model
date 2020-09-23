# coding:utf8
import tqdm
import os
import time
from collections import defaultdict
import json
import sys
import pdb
import traceback
from bdb import BdbQuit
import torch
import kenlm
import pickle
import random

from opts import parse_opt
from models.decoder import Detector
from models.sent_senti_cls import SentenceSentimentClassifier
from dataloader import get_rl_fact_dataloader, get_rl_senti_dataloader, get_senti_corpus_with_sentis_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    dataset_name = opt.dataset_name
    corpus_type = opt.corpus_type

    idx2word = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'idx2word.json'), 'r'))
    img_captions = json.load(open(os.path.join(opt.captions_dir, dataset_name, 'img_captions.json'), 'r'))
    img_det_concepts = json.load(open(os.path.join(opt.captions_dir, dataset_name, 'img_det_concepts.json'), 'r'))
    img_det_sentiments = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'img_det_sentiments.json'), 'r'))
    img_senti_labels = json.load(open(opt.img_senti_labels, 'r'))
    senti_captions = json.load(open(os.path.join(opt.captions_dir, dataset_name, corpus_type, 'senti_captions.json'), 'r'))
    sentiment_words = json.load(open(os.path.join(opt.corpus_dir, corpus_type, 'sentiment_words.json'), 'r'))

    model = Detector(idx2word, opt.max_seq_len, opt.sentiment_categories, opt.rl_lrs, opt.settings)
    model.to(opt.device)
    if opt.rl_resume:
        print("====> loading checkpoint '{}'".format(opt.rl_resume))
        chkpoint = torch.load(opt.rl_resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        assert opt.max_seq_len == chkpoint['max_seq_len'], \
            'opt.max_seq_len and resume model max_seq_len are different'
        assert opt.sentiment_categories == chkpoint['sentiment_categories'], \
            'opt.sentiment_categories and resume model sentiment_categories are different'
        assert dataset_name == chkpoint['dataset_name'], \
            'dataset_name and resume model dataset_name are different'
        assert corpus_type == chkpoint['corpus_type'], \
            'corpus_type and resume model corpus_type are different'
        model.load_state_dict(chkpoint['model'])
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.rl_resume, chkpoint['epoch']))
    else:
        rl_xe_resume = os.path.join(opt.checkpoint, 'xe', dataset_name, corpus_type, 'model-best.pth')
        print("====> loading checkpoint '{}'".format(rl_xe_resume))
        chkpoint = torch.load(rl_xe_resume, map_location=lambda s, l: s)
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
        model.captioner.load_state_dict(chkpoint['model'])
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(rl_xe_resume, chkpoint['epoch']))

        if opt.rl_senti_resume:
            print("====> loading rl_senti_resume '{}'".format(opt.rl_senti_resume))
            ch = torch.load(opt.rl_senti_resume, map_location=lambda s, l: s)
            assert opt.settings == ch['settings'], \
                'opt.settings and rl_senti_resume settings are different'
            assert opt.sentiment_categories == ch['sentiment_categories'], \
                'opt.sentiment_categories and rl_senti_resume sentiment_categories are different'
            model.senti_detector.load_state_dict(ch['model'])

        if True:
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
            model.sent_senti_cls.load_state_dict(chkpoint['model'])

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
                tmp.append([model.captioner.sos_id] +
                           [word2idx.get(w, None) or word2idx['<UNK>'] for w in seq] +
                           [model.captioner.eos_id])
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

    senti_label2idx = {}
    for i, w in enumerate(opt.sentiment_categories):
        senti_label2idx[w] = i
    print('====> process image senti_labels begin')
    senti_labels_id = {}
    for split, senti_labels in img_senti_labels.items():
        print('convert %s senti_labels to index' % split)
        senti_labels_id[split] = []
        for fn, senti_label in tqdm.tqdm(senti_labels):
            senti_labels_id[split].append([fn, senti_label2idx[senti_label]])
    img_senti_labels = senti_labels_id
    print('====> process image senti_labels end')

    print('====> process senti corpus begin')
    senti_captions['positive'] = senti_captions['positive'] * int(len(senti_captions['neutral']) / len(senti_captions['positive']))
    senti_captions['negative'] = senti_captions['negative'] * int(len(senti_captions['neutral']) / len(senti_captions['negative']))
    senti_captions_id = []
    for senti, caps in senti_captions.items():
        print('convert %s corpus to index' % senti)
        senti_id = senti_label2idx[senti]
        for cap, cpts, sentis in tqdm.tqdm(caps):
            cap = [model.captioner.sos_id] +\
                  [word2idx.get(w, None) or word2idx['<UNK>'] for w in cap] +\
                  [model.captioner.eos_id]
            cpts = [word2idx[w] for w in cpts if w in word2idx]
            sentis = [word2idx[w] for w in sentis]
            senti_captions_id.append([cap, cpts, sentis, senti_id])
    random.shuffle(senti_captions_id)
    senti_captions = senti_captions_id
    print('====> process senti corpus end')

    print('====> process sentiment words begin')
    tmp_sentiment_words = {}
    for senti in opt.sentiment_categories:
        senti_id = senti_label2idx[senti]
        if senti not in sentiment_words:
            tmp_sentiment_words[senti_id] = dict()
        else:
            tmp_sentiment_words[senti_id] = {word2idx[w]: 1.0 for w, s in sentiment_words[senti].items()}
    sentiment_words = tmp_sentiment_words
    print('====> process sentiment words end')

    fc_feats = os.path.join(opt.feats_dir, dataset_name, '%s_fc.h5' % dataset_name)
    att_feats = os.path.join(opt.feats_dir, dataset_name, '%s_att.h5' % dataset_name)
    fact_train_data = get_rl_fact_dataloader(
        fc_feats, att_feats, img_captions['train'], img_det_concepts,
        img_det_sentiments, model.captioner.pad_id, opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works)
    fact_val_data = get_rl_fact_dataloader(
        fc_feats, att_feats, img_captions['val'], img_det_concepts,
        img_det_sentiments, model.captioner.pad_id, opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works, shuffle=False)
    test_captions = {}
    for fn in img_captions['test']:
        test_captions[fn] = [[]]
    fact_test_data = get_rl_fact_dataloader(
        fc_feats, att_feats, test_captions, img_det_concepts,
        img_det_sentiments, model.captioner.pad_id, opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works, shuffle=False)

    senti_fc_feats = os.path.join(opt.feats_dir, 'sentiment', 'feats_fc.h5')
    senti_att_feats = os.path.join(opt.feats_dir, 'sentiment', 'feats_att.h5')
    senti_train_data = get_rl_senti_dataloader(
        senti_fc_feats, senti_att_feats, img_det_concepts,
        img_det_sentiments, img_senti_labels['train'], model.captioner.pad_id,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works)
    senti_val_data = get_rl_senti_dataloader(
        senti_fc_feats, senti_att_feats, img_det_concepts,
        img_det_sentiments, img_senti_labels['val'], model.captioner.pad_id,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works, shuffle=False)
    senti_test_data = get_rl_senti_dataloader(
        senti_fc_feats, senti_att_feats, img_det_concepts,
        img_det_sentiments, img_senti_labels['test'], model.captioner.pad_id,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works, shuffle=False)

    scs_data = get_senti_corpus_with_sentis_dataloader(
        senti_captions, idx2word.index('<PAD>'), opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, 80, opt.rl_num_works)

    # lms = {}
    # lm_dir = os.path.join(opt.captions_dir, dataset_name, corpus_type, 'lm')
    # for senti, i in senti_label2idx.items():
    #     lms[i] = kenlm.LanguageModel(os.path.join(lm_dir, '%s_id.kenlm.arpa' % senti))
    # model.set_lms(lms)

    model.set_ciderd_scorer(img_captions)
    model.set_sentiment_words(sentiment_words)

    tmp_dir = ''
    checkpoint = os.path.join(opt.checkpoint, 'rl', dataset_name, corpus_type, tmp_dir)
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    result_dir = os.path.join(opt.result_dir, 'rl', dataset_name, corpus_type, tmp_dir)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    for epoch in range(opt.rl_epochs):
        print('--------------------epoch: %d' % epoch)
        print('tmp_dir:', tmp_dir, 'cls_flag:', model.cls_flag, 'seq_flag:', model.seq_flag)
        torch.cuda.empty_cache()
        for i in range(opt.rl_senti_times):
            print('----------rl_senti_times: %d' % i)
            senti_train_loss = model((senti_train_data, scs_data), data_type='senti', training=True)
            print('senti_train_loss: %s' % dict(senti_train_loss))
        for i in range(opt.rl_fact_times):
            # seq 在前面比较好
            # print('----------seq2seq train')
            # seq2seq_train_loss = model(scs_data, data_type='seq2seq', training=True)
            # print('seq2seq_train_loss: %s' % seq2seq_train_loss)
            print('----------rl_fact_times: %d' % i)
            fact_train_loss = model((fact_train_data, scs_data), data_type='fact', training=True)
            print('fact_train_loss: %s' % dict(fact_train_loss))

        with torch.no_grad():
            torch.cuda.empty_cache()
            print('----------val')
            fact_val_loss = model((fact_val_data,), data_type='fact', training=False)
            print('fact_val_loss:', dict(fact_val_loss))

            # test
            results = {'fact': defaultdict(list), 'senti': defaultdict(list)}
            det_sentis = defaultdict(dict)
            senti_imgs_num = 0
            senti_imgs_wrong_num = 0
            for data_type, data in [('fact', fact_test_data), ('senti', senti_test_data)]:
                print('----------test:', data_type)
                for data_item in tqdm.tqdm(data):
                    if data_type == 'fact':
                        fns, fc_feats, att_feats, _, cpts_tensor, sentis_tensor, ground_truth = data_item
                    elif data_type == 'senti':
                        fns, fc_feats, att_feats, cpts_tensor, sentis_tensor, senti_labels = data_item
                        senti_labels = senti_labels.to(opt.device)
                        senti_labels = [opt.sentiment_categories[int(idx)] for idx in senti_labels]
                    else:
                        raise Exception('data_type(%s) is wrong!' % data_type)
                    fc_feats = fc_feats.to(opt.device)
                    att_feats = att_feats.to(opt.device)
                    sentis_tensor = sentis_tensor.to(opt.device)

                    for i, fn in enumerate(fns):
                        captions, det_img_sentis = model.sample(
                            fc_feats[i], att_feats[i], sentis_tensor[i], beam_size=opt.beam_size)
                        results[data_type][det_img_sentis[0]].append({'image_id': fn, 'caption': captions[0]})
                        det_sentis[data_type][fn] = det_img_sentis[0]
                        if data_type == 'senti':
                            senti_imgs_num += 1
                            if det_img_sentis[0] != senti_labels[i]:
                                senti_imgs_wrong_num += 1

            det_sentis_wrong_rate = senti_imgs_wrong_num / senti_imgs_num

            for data_type in results:
                for senti in results[data_type]:
                    json.dump(results[data_type][senti],
                              open(os.path.join(result_dir, 'result_%d_%s_%s.json' % (epoch, senti, data_type)), 'w'))
                wr = det_sentis_wrong_rate
                if data_type == 'fact':
                    wr = 0
                json.dump(det_sentis[data_type],
                          open(os.path.join(result_dir, 'result_%d_sentis_%s_%s.json' % (epoch, wr, data_type)), 'w'))

            sents = {'fact': defaultdict(str), 'senti': defaultdict(str)}
            sents_w = {'fact': defaultdict(str), 'senti': defaultdict(str)}
            for data_type in results:
                for senti in results[data_type]:
                    ress = results[data_type][senti]
                    for res in ress:
                        caption = res['caption']
                        sents_w[data_type][senti] += caption + '\n'
                        caption = [str(word2idx[w]) for w in caption.split()] + [str(word2idx['<EOS>'])]
                        caption = ' '.join(caption) + '\n'
                        sents[data_type][senti] += caption
            for data_type in sents:
                for senti in sents[data_type]:
                    with open(os.path.join(result_dir, 'result_%d_%s_%s.txt' % (epoch, senti, data_type)), 'w') as f:
                        f.write(sents[data_type][senti])
                    with open(os.path.join(result_dir, 'result_%d_%s_%s_w.txt' % (epoch, senti, data_type)), 'w') as f:
                        f.write(sents_w[data_type][senti])

        if epoch > -1:
            chkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'max_seq_len': opt.max_seq_len,
                'sentiment_categories': opt.sentiment_categories,
                'dataset_name': dataset_name,
                'corpus_type': corpus_type,
            }
            checkpoint_path = os.path.join(
                checkpoint, 'model_%d_%s.pth' % (
                    epoch, time.strftime('%m%d-%H%M')))
            torch.save(chkpoint, checkpoint_path)


if __name__ == '__main__':
    try:
        opt = parse_opt()
        train()
    except (BdbQuit, torch.cuda.memory_allocated()):
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
