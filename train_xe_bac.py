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
from torch.nn.utils.rnn import pack_padded_sequence

from opts import parse_opt
from models.captioner import Captioner
from dataloader import get_caption_dataloader


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def train():
    idx2word = json.load(open(opt.idx2word, 'r'))
    img_captions = json.load(open(opt.img_captions, 'r'))
    img_det_concepts = json.load(open(opt.img_det_concepts, 'r'))

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
                tmp.append([word2idx['<SOS>']] +
                           [word2idx.get(w, None) or word2idx['<UNK>'] for w in seq] +
                           [word2idx['<EOS>']])
            captions_id[split][fn] = tmp
    img_captions = captions_id
    print('====> process image captions end')

    print('====> process image det_concepts begin')
    det_concepts_id = {}
    for fn, cpts in tqdm.tqdm(img_det_concepts.items()):
        det_concepts_id[fn] = [word2idx[w] for w in cpts]
    img_det_concepts = det_concepts_id
    print('====> process image det_concepts end')

    captioner = Captioner(idx2word, opt.settings)
    captioner.to(opt.device)

    lr = opt.xe_lr
    optimizer, criterion = captioner.get_optim_criterion(lr)
    if opt.xe_resume:
        print("====> loading checkpoint '{}'".format(opt.xe_resume))
        chkpoint = torch.load(opt.xe_resume, map_location=lambda s, l: s)
        assert opt.settings == chkpoint['settings'], \
            'opt.settings and resume model settings are different'
        assert idx2word == chkpoint['idx2word'], \
            'idx2word and resume model idx2word are different'
        captioner.load_state_dict(chkpoint['model'])
        optimizer.load_state_dict(chkpoint['optimizer'])
        lr = optimizer.param_groups[0]['lr']
        print("====> loaded checkpoint '{}', epoch: {}"
              .format(opt.xe_resume, chkpoint['epoch']))

    train_data = get_caption_dataloader(opt.fc_feats, opt.att_feats, img_captions['train'],
                                        img_det_concepts, idx2word.index('<PAD>'),
                                        opt.max_seq_len, opt.num_concepts,
                                        opt.xe_bs, opt.xe_num_works)
    val_data = get_caption_dataloader(opt.fc_feats, opt.att_feats, img_captions['val'],
                                      img_det_concepts, idx2word.index('<PAD>'),
                                      opt.max_seq_len, opt.num_concepts, opt.xe_bs,
                                      opt.xe_num_works, shuffle=False)
    test_data = get_caption_dataloader(opt.fc_feats, opt.att_feats, img_captions['test'],
                                       img_det_concepts, idx2word.index('<PAD>'),
                                       opt.max_seq_len, opt.num_concepts, opt.xe_bs,
                                       opt.xe_num_works, shuffle=False)

    def forward(data, training=True, ss_prob=0.0):
        captioner.train(training)
        loss_val = 0.0
        for _, fc_feats, att_feats, (caps_tensor, lengths), cpts_tensor in tqdm.tqdm(data):
            fc_feats = fc_feats.to(opt.device)
            att_feats = att_feats.to(opt.device)
            caps_tensor = caps_tensor.to(opt.device)
            cpts_tensor = cpts_tensor.to(opt.device)

            pred = captioner(fc_feats, att_feats, cpts_tensor, caps_tensor,
                             lengths, ss_prob, mode='xe')
            real = pack_padded_sequence(caps_tensor[:, 1:], lengths, batch_first=True)[0]
            loss = criterion(pred, real)
            loss_val += loss.item()
            if training:
                optimizer.zero_grad()
                loss.backward()
                # TODO
                clip_gradient(optimizer, opt.grad_clip)
                optimizer.step()
        return loss_val / len(data)

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
        ss_prob = 0.0
        if epoch > opt.scheduled_sampling_start >= 0:
            frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
            ss_prob = min(opt.scheduled_sampling_increase_prob * frac, opt.scheduled_sampling_max_prob)
        train_loss = forward(train_data, ss_prob=ss_prob)
        with torch.no_grad():
            val_loss = forward(val_data, training=False)

            results = []
            fact_txt = ''
            for fns, fc_feats, att_feats, _, cpts_tensor in tqdm.tqdm(test_data):
                fc_feats = fc_feats.to(opt.device)
                att_feats = att_feats.to(opt.device)
                cpts_tensor = cpts_tensor.to(opt.device)
                for i, fn in enumerate(fns):
                    captions, _ = captioner.sample(
                        fc_feats[i], att_feats[i], cpts_tensor[i],
                        beam_size=opt.beam_size, max_seq_lenth=opt.max_seq_len)
                    results.append({'image_id': fn, 'caption': captions[0]})
                    fact_txt += captions[0] + '\n'
            json.dump(results, open(os.path.join(result_dir, 'result_%d.json' % epoch), 'w'))
            with open(os.path.join(result_dir, 'result_%d.txt' % epoch), 'w') as f:
                f.write(fact_txt)

        if previous_loss is not None and val_loss > previous_loss:
            lr = lr * 0.5
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        previous_loss = val_loss

        print('train_loss: %.4f, val_loss: %.4f' % (train_loss, val_loss))
        if epoch in [0, 10, 15, 20, 25, 29, 30, 35, 39]:
            chkpoint = {
                'epoch': epoch,
                'model': captioner.state_dict(),
                'optimizer': optimizer.state_dict(),
                'settings': opt.settings,
                'idx2word': idx2word,
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
