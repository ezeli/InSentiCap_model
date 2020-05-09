# coding:utf8
import torch
import h5py
import json
import tqdm

from opts import parse_opt
from models.sentiment_detector import SentimentDetector
from dataloader import get_senti_image_dataloader


opt = parse_opt()
f_senti_att = h5py.File(opt.senti_att_feats, mode='r')
img_senti_labels = json.load(open(opt.img_senti_labels, 'r'))

print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = torch.load(opt.eval_model, map_location=lambda s, l: s)
sentiment_categories = chkpoint['sentiment_categories']
model = SentimentDetector(sentiment_categories, chkpoint['settings'])
model.to(opt.device)
model.load_state_dict(chkpoint['model'])
model.eval()
print("====> loaded checkpoint '{}', epoch: {}".
      format(opt.eval_model, chkpoint['epoch']))

split = 'test'
test_data = get_senti_image_dataloader(
    f_senti_att, img_senti_labels[split],sentiment_categories,
    opt.senti_bs, shuffle=False)

corr_rate = 0.0
for _, att_feats, labels in tqdm.tqdm(test_data):
    att_feats = att_feats.to(opt.device)
    labels = labels.to(opt.device)
    idx, _, _, _ = model.sample(att_feats)

    corr = sum(labels == idx)
    corr_rate += corr / len(idx)

data_len = len(test_data)
print('Correct rate: %s' % (corr_rate / data_len))



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
