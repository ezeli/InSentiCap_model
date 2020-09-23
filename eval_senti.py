import torch
import json
import os
from collections import defaultdict

from models.sentiment_detector import SentimentDetector
from dataloader import get_senti_image_dataloader
from opts import parse_opt


labeled_file = './data/labeled_data/at_most_one_disagree.json'
labeled_data = json.load(open(labeled_file, 'r'))

opt = parse_opt()
print("====> loading rl_senti_resume '{}'".format(opt.rl_senti_resume))
ch = torch.load(opt.rl_senti_resume, map_location=lambda s, l: s)
settings = ch['settings']
sentiment_categories = ch['sentiment_categories']
model = SentimentDetector(sentiment_categories, settings)
model.load_state_dict(ch['model'])
model.to(opt.device)
model.eval()

senti_label2idx = {}
for i, w in enumerate(sentiment_categories):
    senti_label2idx[w] = i
neu_idx = senti_label2idx['neutral']
img_senti_labels = {}
for senti, fns in labeled_data.items():
    senti_id = senti_label2idx[senti]
    img_senti_labels[senti] = [[fn, senti_id] for fn in fns]

dataset_name = 'coco'
att_feats = os.path.join(opt.feats_dir, dataset_name, '%s_att.h5' % dataset_name)
eval_datas = {}
for senti in img_senti_labels:
    data = get_senti_image_dataloader(
        att_feats, img_senti_labels[senti], batch_size=len(img_senti_labels[senti]),
        num_workers=2, shuffle=False)
    eval_datas[senti] = next(iter(data))

for THRESHOLD in range(11):
    THRESHOLD = THRESHOLD / 10
    print('THRESHOLD:', THRESHOLD)
    all_num = 0
    all_cor_num = 0
    for senti, (_, att_feats, labels) in eval_datas.items():
        att_feats = att_feats.to(opt.device)
        labels = labels.to(opt.device)
        with torch.no_grad():
            preds, _, _, scores = model.sample(att_feats)
        replace_idx = (scores < THRESHOLD).nonzero(as_tuple=False).view(-1)
        preds.index_copy_(0, replace_idx, preds.new_zeros(len(replace_idx)).fill_(neu_idx))
        num = int(preds.size(0))
        cor_num = int(sum(preds == labels))
        print('%s accuracy: %s' % (senti, cor_num / num))
        # print('%s scores mean: %s' % (senti, scores.mean()))
        all_num += num
        all_cor_num += cor_num
    print('all accuracy:', all_cor_num / all_num)


for THRESHOLD in range(10):
    THRESHOLD = THRESHOLD / 10
    print('THRESHOLD:', THRESHOLD)
    all_num = defaultdict(int)
    all_cor_num = defaultdict(int)
    for senti, (_, att_feats, labels) in eval_datas.items():
        att_feats = att_feats.to(opt.device)
        labels = labels.to(opt.device)
        with torch.no_grad():
            preds, _, _, scores = model.sample(att_feats)
        replace_idx = (scores < THRESHOLD).nonzero(as_tuple=False).view(-1)
        preds.index_copy_(0, replace_idx, preds.new_zeros(len(replace_idx)).fill_(neu_idx))
        for idx in [0, 1, 2]:
            all_num[idx] += int(sum(preds == idx))
        label = int(labels[0])
        all_cor_num[label] += int(sum(preds == label))
    for senti_id in all_num:
        senti = sentiment_categories[senti_id]
        print('%s precision: %s' % (senti, all_cor_num[senti_id] / all_num[senti_id]))
    print('all precision:', sum(all_cor_num.values()) / sum(all_num.values()))
    for senti_id in all_num:
        senti = sentiment_categories[senti_id]
        print('%s all num: %s, cor num: %s' % (senti, all_num[senti_id], all_cor_num[senti_id]))
