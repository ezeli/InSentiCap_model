# coding:utf8
import torch
from copy import deepcopy
import json
import tqdm
import numpy as np

from opts import parse_opt
from models.sent_senti_cls import SentenceSentimentClassifier
from dataloader import get_senti_sents_dataloader

opt = parse_opt()
assert opt.eval_model, 'please input eval_model'

print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = torch.load(opt.eval_model, map_location=lambda s, l: s)
idx2word = chkpoint['idx2word']
sentiment_categories = chkpoint['sentiment_categories']
settings = chkpoint['settings']
model = SentenceSentimentClassifier(idx2word, sentiment_categories, settings)
model.load_state_dict(chkpoint['model'])
print("====> loaded checkpoint '{}', epoch: {}".
      format(opt.eval_model, chkpoint['epoch']))
model.to(opt.device)
model.eval()

senti_corpus = json.load(open(opt.real_captions, 'r'))['senti']
word2idx = {}
for i, w in enumerate(idx2word):
    word2idx[w] = i
senti_label2idx = {}
for i, w in enumerate(sentiment_categories):
    senti_label2idx[w] = i

print('====> process senti_corpus begin')
val_sets = {}
val_sets['all'] = []
for senti in opt.sentiment_categories:
    print('convert %s corpus to index' % senti)
    senti_id = senti_label2idx[senti]
    val_sets[senti] = []
    sents = deepcopy(senti_corpus[senti])
    for sent in sents:
        tmp = [word2idx.get(w, None) or word2idx['<UNK>'] for w in sent] + [word2idx['<EOS>']]
        val_sets[senti].append([senti_id, tmp])
        val_sets['all'].append([senti_id, tmp])
print('====> process senti_corpus end')

val_datas = {}
for senti in val_sets:
    val_datas[senti] = get_senti_sents_dataloader(val_sets[senti], word2idx['<PAD>'], opt.max_seq_len, shuffle=False)

for senti, val_data in val_datas.items():
    all_num = 0
    wrong_num = 0
    with torch.no_grad():
        for sentis, (caps_tensor, lengths) in tqdm.tqdm(val_data):
            sentis = sentis.to(opt.device)
            caps_tensor = caps_tensor.to(opt.device)

            rest, _ = model.sample(caps_tensor, lengths)
            rest = torch.LongTensor(np.array(rest)).to(opt.device)
            all_num += int(sentis.size(0))
            wrong_num += int((sentis != rest).sum())
    wrong_rate = wrong_num / all_num
    print('%s wrong_rate: %.6f' % (senti, wrong_rate))
