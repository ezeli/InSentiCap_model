# coding:utf8
import torch
import json
import tqdm
import os
import h5py
import numpy as np

from opts import parse_opt
from models.concept_detector import ConceptDetector
from dataloader import get_concept_dataloader


opt = parse_opt()
print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = torch.load(opt.eval_model, map_location=lambda s, l: s)
idx2concept = chkpoint['idx2concept']
settings = chkpoint['settings']
dataset_name = chkpoint['dataset_name']
model = ConceptDetector(idx2concept, settings)
model.to(opt.device)
model.load_state_dict(chkpoint['model'])
model.eval()
_, criterion = model.get_optim_criterion(0)
print("====> loaded checkpoint '{}', epoch: {}, dataset_name: {}".
      format(opt.eval_model, chkpoint['epoch'], dataset_name))


fact_fc = h5py.File(os.path.join(opt.feats_dir, dataset_name, '%s_fc.h5' % dataset_name), 'r')
senti_fc = h5py.File(os.path.join(opt.feats_dir, 'sentiment', 'feats_fc.h5'), 'r')

predict_result = {}
for fc in [fact_fc, senti_fc]:
    fns = list(fc.keys())
    for i in tqdm.tqdm(range(0, len(fns), 100)):
        cur_fns = fns[i:i + 100]
        feats = []
        for fn in cur_fns:
            feats.append(fc[fn][:])
        feats = torch.FloatTensor(np.array(feats)).to(opt.device)
        _, concepts, _ = model.sample(feats, num=opt.num_concepts)
        for j, fn in enumerate(cur_fns):
            predict_result[fn] = concepts[j]

json.dump(predict_result, open(os.path.join(opt.captions_dir, dataset_name, 'img_det_concepts.json'), 'w'))
