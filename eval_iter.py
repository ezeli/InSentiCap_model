# coding:utf8
import torch
import os
import json
import tqdm
from collections import defaultdict

from opts import parse_opt
from models.insenti_cap import InSentiCap
from dataloader import get_iter_fact_dataloader, get_iter_senti_dataloader

opt = parse_opt()
assert opt.eval_model, 'please input eval_model'
assert opt.result_file, 'please input result_file'

img_captions = json.load(open(opt.img_captions, 'r'))
img_det_concepts = json.load(open(opt.img_det_concepts, 'r'))
img_det_sentiments = json.load(open(opt.img_det_sentiments, 'r'))
img_senti_labels = json.load(open(opt.img_senti_labels, 'r'))

print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = torch.load(opt.eval_model, map_location=lambda s, l: s)
epoch = chkpoint['epoch']
idx2word = chkpoint['idx2word']
model = InSentiCap(idx2word, chkpoint['max_seq_len'],
                   chkpoint['sentiment_categories'], defaultdict(int),
                   chkpoint['iter_hyperparams'], None, chkpoint['settings'])
model.load_state_dict(chkpoint['model'])
print("====> loaded checkpoint '{}', epoch: {}".
      format(opt.eval_model, epoch))
model.to(opt.device)
model.eval()

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

senti = 'fact'
split = 'test'
test_data = get_iter_fact_dataloader(
    opt.fc_feats, opt.att_feats, img_det_concepts, img_det_sentiments,
    img_captions[split].keys(), idx2word.index('<PAD>'),
    opt.num_concepts, opt.num_sentiments, opt.iter_bs, opt.iter_num_works, shuffle=False)

results = []
det_sentis = {}
for fns, fc_feats, att_feats, cpts_tensor, sentis_tensor in \
        tqdm.tqdm(test_data):
    fc_feats = fc_feats.to(opt.device)
    att_feats = att_feats.to(opt.device)
    cpts_tensor = cpts_tensor.to(opt.device)
    sentis_tensor = sentis_tensor.to(opt.device)

    for i, fn in enumerate(fns):
        captions, det_img_sentis = model.sample(
            fc_feats[i], att_feats[i], cpts_tensor[i],
            sentis_tensor[i], beam_size=opt.beam_size)
        results.append({'image_id': fn, 'caption': captions[0]})
        det_sentis[fn] = det_img_sentis[0]

result_dir = './result'
json.dump(results, open(os.path.join(result_dir, 'result_%d.json' % epoch), 'w'))
json.dump(det_sentis, open(os.path.join(result_dir, 'result_%d_sentis.json' % epoch), 'w'))
