# coding:utf8
import os
import tqdm
from collections import defaultdict
import time
import json
import sys
import pdb
import traceback
from bdb import BdbQuit
import torch

from opts import parse_opt
from models.insenti_cap import InSentiCap
from dataloader import get_iter_fact_dataloader, get_iter_senti_dataloader


# def clip_gradient(optimizer, grad_clip):
#     for group in optimizer.param_groups:
#         for param in group['params']:
#             param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    idx2word = json.load(open(opt.idx2word, 'r'))
    img_captions = json.load(open(opt.img_captions, 'r'))
    img_det_concepts = json.load(open(opt.img_det_concepts, 'r'))
    img_det_sentiments = json.load(open(opt.img_det_sentiments, 'r'))
    img_senti_labels = json.load(open(opt.img_senti_labels, 'r'))
    real_captions = json.load(open(opt.real_captions, 'r'))

    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i

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

    print('====> process real_captions begin')
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
    print('====> process real_captions end')

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

    fact_train_data = get_iter_fact_dataloader(
        opt.fc_feats, opt.att_feats, img_captions['train'], img_det_concepts, img_det_sentiments,
        idx2word.index('<PAD>'), opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.iter_bs, opt.iter_num_works)
    fact_val_data = get_iter_fact_dataloader(
        opt.fc_feats, opt.att_feats, img_captions['val'], img_det_concepts, img_det_sentiments,
        idx2word.index('<PAD>'), opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.iter_bs, opt.iter_num_works, shuffle=False)
    fact_test_data = get_iter_fact_dataloader(
        opt.fc_feats, opt.att_feats, img_captions['test'], img_det_concepts, img_det_sentiments,
        idx2word.index('<PAD>'), opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.iter_bs, opt.iter_num_works, shuffle=False)
    del img_captions

    senti_train_data = get_iter_senti_dataloader(
        opt.senti_fc_feats, opt.senti_att_feats, img_det_concepts, img_det_sentiments,
        img_senti_labels['train'], idx2word.index('<PAD>'),
        opt.num_concepts, opt.num_sentiments, opt.iter_bs, opt.iter_num_works)
    senti_val_data = get_iter_senti_dataloader(
        opt.senti_fc_feats, opt.senti_att_feats, img_det_concepts, img_det_sentiments,
        img_senti_labels['val'], idx2word.index('<PAD>'),
        opt.num_concepts, opt.num_sentiments, opt.iter_bs, opt.iter_num_works, shuffle=False)

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
        assert opt.max_sql_len == chkpoint['max_seq_length'], \
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
        if opt.iter_senti_resume:
            print("====> loading iter_senti_resume '{}'".format(opt.iter_senti_resume))
            ch = torch.load(opt.iter_senti_resume, map_location=lambda s, l: s)
            assert opt.settings == ch['settings'], \
                'opt.settings and iter_senti_resume settings are different'
            assert opt.sentiment_categories == ch['sentiment_categories'], \
                'opt.sentiment_categories and iter_senti_resume sentiment_categories are different'
            in_senti_cap.senti_detector.load_state_dict(ch['model'])
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.iter_xe_resume, chkpoint['epoch']))
    else:
        raise Exception('iter_resume or iter_xe_resume is required!')

    checkpoint = os.path.join(opt.checkpoint, 'iter')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    result_dir = './result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    previous_loss = None
    for epoch in range(opt.iter_epochs):
        print('--------------------epoch: %d' % epoch)
        torch.cuda.empty_cache()
        for i in range(opt.iter_senti_times):
            print('----------iter_senti_times: %d' % i)
            senti_train_loss = in_senti_cap(senti_train_data, data_type='senti', training=True)
            print('senti_train_loss: %s' % senti_train_loss)
        for i in range(opt.iter_fact_times):
            print('----------iter_fact_times: %d' % i)
            fact_train_loss = in_senti_cap(fact_train_data, data_type='fact', training=True)
            print('fact_train_loss: %s' % fact_train_loss)
        # torch.save(in_senti_cap.state_dict(), os.path.join(checkpoint, 'in_senti_cap_%s.pth' % epoch))
        with torch.no_grad():
            torch.cuda.empty_cache()
            print('----------val')
            senti_val_loss = in_senti_cap(senti_val_data, data_type='senti', training=False)
            print('senti_val_loss: %s' % senti_val_loss)
            fact_val_loss = in_senti_cap(fact_val_data, data_type='fact', training=False)
            print('fact_val_loss: %s' % fact_val_loss)

            # test
            print('----------test')
            results = []
            det_sentis = {}
            for fns, fc_feats, att_feats, cpts_tensor, sentis_tensor in \
                    tqdm.tqdm(fact_test_data):
                fc_feats = fc_feats.to(opt.device)
                att_feats = att_feats.to(opt.device)
                cpts_tensor = cpts_tensor.to(opt.device)
                sentis_tensor = sentis_tensor.to(opt.device)

                for i, fn in enumerate(fns):
                    captions, det_img_sentis = in_senti_cap.sample(
                        fc_feats[i], att_feats[i], cpts_tensor[i],
                        sentis_tensor[i], beam_size=opt.beam_size)
                    results.append({'image_id': fn, 'caption': captions[0]})
                    det_sentis[fn] = det_img_sentis[0]

            json.dump(results, open(os.path.join(result_dir, 'result_%d.json' % epoch), 'w'))
            json.dump(det_sentis, open(os.path.join(result_dir, 'result_%d_sentis.json' % epoch), 'w'))

            sents = defaultdict(str)
            for res in results:
                fn = res['image_id']
                caption = res['caption'].split()
                caption = [str(word2idx[w]) for w in caption] + [str(word2idx['<EOS>'])]
                caption = ' '.join(caption) + '\n'
                sents[det_sentis[fn]] += caption
            for senti in sents:
                with open(os.path.join(result_dir, 'result_%d_%s.txt' % (epoch, senti)), 'w') as f:
                    f.write(sents[senti])

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
        if epoch in [0, 1, 2, 3, 5, 7, 10, 13, 15, 17, 20, 23, 25, 27, 29]:
            chkpoint = {
                'epoch': epoch,
                'model': in_senti_cap.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'max_seq_length': opt.max_sql_len,
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
