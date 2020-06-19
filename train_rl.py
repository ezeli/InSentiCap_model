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
import kenlm

from opts import parse_opt
from models.decoder import Detector
from dataloader import get_rl_fact_dataloader, get_rl_senti_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    idx2word = json.load(open(opt.idx2word, 'r'))
    img_captions = json.load(open(opt.img_captions, 'r'))
    img_det_concepts = json.load(open(opt.img_det_concepts, 'r'))
    img_det_sentiments = json.load(open(opt.img_det_sentiments, 'r'))
    img_senti_labels = json.load(open(opt.img_senti_labels, 'r'))

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
        model.load_state_dict(chkpoint['model'])
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.rl_resume, chkpoint['epoch']))
    elif opt.rl_xe_resume:
        print("====> loading checkpoint '{}'".format(opt.rl_xe_resume))
        chkpoint = torch.load(opt.rl_xe_resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        model.captioner.load_state_dict(chkpoint['model'])
        if opt.rl_senti_resume:
            print("====> loading iter_senti_resume '{}'".format(opt.rl_senti_resume))
            ch = torch.load(opt.rl_senti_resume, map_location=lambda s, l: s)
            assert opt.settings == ch['settings'], \
                'opt.settings and rl_senti_resume settings are different'
            assert opt.sentiment_categories == ch['sentiment_categories'], \
                'opt.sentiment_categories and rl_senti_resume sentiment_categories are different'
            model.senti_detector.load_state_dict(ch['model'])
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.rl_xe_resume, chkpoint['epoch']))
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

    fact_train_data = get_rl_fact_dataloader(
        opt.fc_feats, opt.att_feats, img_captions['train'], img_det_concepts,
        img_det_sentiments, model.captioner.pad_id, opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works)
    fact_val_data = get_rl_fact_dataloader(
        opt.fc_feats, opt.att_feats, img_captions['val'], img_det_concepts,
        img_det_sentiments, model.captioner.pad_id, opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works, shuffle=False)
    fact_test_data = get_rl_fact_dataloader(
        opt.fc_feats, opt.att_feats, img_captions['test'], img_det_concepts,
        img_det_sentiments, model.captioner.pad_id, opt.max_seq_len,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works, shuffle=False)

    senti_train_data = get_rl_senti_dataloader(
        opt.senti_fc_feats, opt.senti_att_feats, img_det_concepts,
        img_det_sentiments, img_senti_labels['train'], model.captioner.pad_id,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works)
    senti_val_data = get_rl_senti_dataloader(
        opt.senti_fc_feats, opt.senti_att_feats, img_det_concepts,
        img_det_sentiments, img_senti_labels['val'], model.captioner.pad_id,
        opt.num_concepts, opt.num_sentiments, opt.rl_bs, opt.rl_num_works, shuffle=False)

    model.set_ciderd_scorer(img_captions)
    lms = {}
    for senti, i in senti_label2idx.items():
        lms[i] = kenlm.LanguageModel(os.path.join(opt.lm_dir, '%s.arpa' % senti))
    model.set_lms(lms)

    checkpoint = os.path.join(opt.checkpoint, 'rl')
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    result_dir = './result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    previous_loss = None
    for epoch in range(opt.rl_epochs):
        print('--------------------epoch: %d' % epoch)
        torch.cuda.empty_cache()
        for i in range(opt.rl_senti_times):
            print('----------rl_senti_times: %d' % i)
            senti_train_loss = model(senti_train_data, data_type='senti', training=True)
            print('senti_train_loss:', senti_train_loss)
        for i in range(opt.rl_fact_times):
            print('----------rl_fact_times: %d' % i)
            fact_train_loss = model(fact_train_data, data_type='fact', training=True)
            print('fact_train_loss:', fact_train_loss)
        # torch.save(model.state_dict(), os.path.join(checkpoint, 'model_%s.pth' % epoch))
        with torch.no_grad():
            torch.cuda.empty_cache()
            print('----------val')
            senti_val_loss = model(senti_val_data, data_type='senti', training=False)
            print('senti_val_loss:', senti_val_loss)
            fact_val_loss = model(fact_val_data, data_type='fact', training=False)
            print('fact_val_loss:', fact_val_loss)

            # test
            results = []
            det_sentis = {}
            for data_item in tqdm.tqdm(fact_test_data):
                fns, fc_feats, att_feats, cpts_tensor, sentis_tensor, _ = data_item
                fc_feats = fc_feats.to(opt.device)
                att_feats = att_feats.to(opt.device)
                cpts_tensor = cpts_tensor.to(opt.device)
                sentis_tensor = sentis_tensor.to(opt.device)
                del data_item

                for i, fn in enumerate(fns):
                    captions, det_img_sentis = model.sample(
                        fc_feats[i], att_feats[i], cpts_tensor[i], sentis_tensor[i],
                        beam_size=opt.beam_size)
                    results.append({'image_id': fn, 'caption': captions[0]})
                    det_sentis[fn] = det_img_sentis[0]

            json.dump(results, open(os.path.join(result_dir, 'result_%d.json' % epoch), 'w'))
            json.dump(det_sentis, open(os.path.join(result_dir, 'result_%d_sentis.json' % epoch), 'w'))

        if previous_loss is not None and senti_val_loss[0] > previous_loss[0] \
                and fact_val_loss[0] > previous_loss[1]:
            for optim in [model.cap_optim, model.senti_optim]:
                lr = optim.param_groups[0]['lr'] * 0.5
                for param_group in optim.param_groups:
                    param_group['lr'] = lr
        previous_loss = [senti_val_loss[0], fact_val_loss[0]]

        if epoch in [0, 1, 2, 3, 5, 7, 9, 10, 12, 14, 15, 17, 19]:
            chkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
                'max_seq_len': opt.max_seq_len,
                'sentiment_categories': opt.sentiment_categories,
            }
            checkpoint_path = os.path.join(
                checkpoint, 'model_%d_%.4f_%.4f_%s.pth' % (
                    epoch, senti_val_loss[0], fact_val_loss[0], time.strftime('%m%d-%H%M')))
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
