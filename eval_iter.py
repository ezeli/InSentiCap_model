# coding:utf8
import torch
import h5py
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
f_fc = h5py.File(opt.fc_feats, mode='r')
f_att = h5py.File(opt.att_feats, mode='r')
f_senti_fc = h5py.File(opt.senti_fc_feats, mode='r')
f_senti_att = h5py.File(opt.senti_att_feats, mode='r')

print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = torch.load(opt.eval_model, map_location=lambda s, l: s)
idx2word = chkpoint['idx2word']
model = InSentiCap(idx2word, chkpoint['max_seq_length'],
                   chkpoint['sentiment_categories'], defaultdict(int),
                   chkpoint['iter_hyperparams'], None, chkpoint['settings'])
model.load_state_dict(chkpoint['model'])
print("====> loaded checkpoint '{}', epoch: {}".
      format(opt.eval_model, chkpoint['epoch']))
model.to(opt.device)
model.eval()

senti = 'fact'
split = 'test'
if senti == 'fact':
    test_data = get_iter_fact_dataloader(
        f_fc, f_att, img_det_concepts, img_det_sentiments, idx2word,
        img_captions[split].keys(), idx2word.index('<PAD>'),
        opt.num_concepts, opt.num_sentiments, opt.iter_bs)
else:
    test_data = get_iter_senti_dataloader(
        f_senti_fc, f_senti_att, img_det_concepts, img_det_sentiments, idx2word,
        img_senti_labels[split], chkpoint['sentiment_categories'], idx2word.index('<PAD>'),
        opt.num_concepts, opt.num_sentiments, opt.iter_bs)

results = []
det_sentis = {}
for data_item in tqdm.tqdm(test_data):
    if senti == 'fact':
        fns, fc_feats, att_feats, cpts_tensor, sentis_tensor = data_item
    else:
        fns, fc_feats, att_feats, cpts_tensor, sentis_tensor, _ = data_item
    fc_feats = fc_feats.to(opt.device)
    att_feats = att_feats.to(opt.device)
    cpts_tensor = cpts_tensor.to(opt.device)
    sentis_tensor = sentis_tensor.to(opt.device)
    del data_item

    for i, fn in enumerate(fns):
        captions, det_img_sentis = model.sample(fc_feats[i], att_feats[i], cpts_tensor[i],
                                                sentis_tensor[i], beam_size=opt.beam_size)
        results.append({'image_id': fn, 'caption': captions[0]})
        det_sentis[fn] = det_img_sentis[0]


json.dump(results, open(opt.result_file, 'w'))
json.dump(det_sentis, open(opt.result_file+'.sentis', 'w'))
