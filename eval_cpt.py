# coding:utf8
import torch
import h5py
import json
import tqdm
import numpy as np

from opts import parse_opt
from models.concept_detector import ConceptDetector
from dataloader import get_concept_dataloader


opt = parse_opt()
img_concepts = json.load(open(opt.img_concepts, 'r'))
f_fc = h5py.File(opt.fc_feats, mode='r')

print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = torch.load(opt.eval_model, map_location=lambda s, l: s)
idx2concept = chkpoint['idx2concept']
settings = chkpoint['settings']
model = ConceptDetector(idx2concept, settings)
model.to(opt.device)
model.load_state_dict(chkpoint['model'])
model.eval()
_, criterion = model.get_optim_criterion(0)
print("====> loaded checkpoint '{}', epoch: {}".
      format(opt.eval_model, chkpoint['epoch']))

predict_result = {}

split = 'train'
test_data = get_concept_dataloader(f_fc, img_concepts[split], idx2concept,
                                   opt.concept_bs, shuffle=False)

loss_val = 0.0
pre = 0.0
recall = 0.0
for fns, fc_feats, cpts_tensors in tqdm.tqdm(test_data):
    fc_feats = fc_feats.to(opt.device)
    cpts_tensors = cpts_tensors.to(opt.device)
    pred, concepts, _ = model.sample(fc_feats, num=opt.num_concepts)
    loss = criterion(pred, cpts_tensors)
    loss_val += loss.item()
    tmp_pre = 0.0
    tmp_rec = 0.0
    for i, fn in enumerate(fns):
        cpts = concepts[i]
        predict_result[fn] = cpts
        grdt = img_concepts[split][fn]
        jiaoji = len(set(grdt) - (set(grdt)-set(cpts)))
        tmp_pre += jiaoji // len(cpts)
        tmp_rec += jiaoji // len(grdt)
    pre += tmp_pre // len(fns)
    recall += tmp_rec // len(fns)


data_len = len(test_data)
print('loss: %s, precision: %s, recall: %s' %
      (loss_val // data_len, pre // data_len, recall // data_len))



# senti_fc = h5py.File(opt.senti_fc_feats, 'r')
# senti_predict_result = {}
# senti_fns = list(senti_fc.keys())
# for i in tqdm.tqdm(range(0, len(senti_fns), 100)):
#     cur_fns = senti_fns[i:i+100]
#     feats = []
#     for fn in cur_fns:
#         feats.append(senti_fc[fn][:])
#     feats = torch.FloatTensor(np.array(feats)).to(opt.device)
#     _, concepts, _ = model.sample(feats, num=opt.num_concepts)
#     for j, fn in enumerate(cur_fns):
#         senti_predict_result[fn] = concepts[j]
# predict_result.update(senti_predict_result)
# json.dump(predict_result, open(opt.img_det_concepts, 'w'))
