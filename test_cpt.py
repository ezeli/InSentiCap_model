# coding:utf8
import torch
import h5py
import json

from opts import parse_opt
from models.concept_detector import ConceptDetector

opt = parse_opt()
img_concepts = json.load(open(opt.img_concepts, 'r'))
f_fc = h5py.File(opt.fc_feats, mode='r')

print("====> loading checkpoint '{}'".format(opt.test_model))
chkpoint = torch.load(opt.test_model, map_location=lambda s, l: s)
idx2concept = chkpoint['idx2concept']
settings = chkpoint['settings']
model = ConceptDetector(idx2concept, settings)
model.to(opt.device)
model.load_state_dict(chkpoint['model'])
model.eval()
print("====> loaded checkpoint '{}', epoch: {}".
      format(opt.test_model, chkpoint['epoch']))

test_img = opt.image_file or list(img_concepts['test'].keys())[0]
feat = torch.FloatTensor(f_fc[test_img][:])
feat = feat.to(opt.device)
feat = feat.unsqueeze(0)
_, concepts, scores = model.sample(feat, num=opt.num_concepts)
concepts = concepts[0]

print('test_img: ', test_img)
print('concepts: ', concepts)
print('scores: ', scores)
print('ground truth: ', img_concepts['test'][test_img])

wrong = []
for c in concepts:
    if c not in img_concepts['test'][test_img]:
        wrong.append(c)
print('\nwrong rate:', len(wrong) / len(concepts))
print('wrong concepts:', wrong)
