import sys
import torch
import tqdm
import numpy as np

from models.sent_senti_cls import SentenceSentimentClassifier
from dataloader import get_senti_sents_dataloader

device = torch.device('cuda:0')
max_seq_len = 16

eval_model = './checkpoint/sent_senti_cls/new/model_15_0.0043_0.0110_0628-1122.pth'
print("====> loading checkpoint '{}'".format(eval_model))
chkpoint = torch.load(eval_model, map_location=lambda s, l: s)
idx2word = chkpoint['idx2word']
sentiment_categories = chkpoint['sentiment_categories']
settings = chkpoint['settings']
model = SentenceSentimentClassifier(idx2word, sentiment_categories, settings)
model.load_state_dict(chkpoint['model'])
print("====> loaded checkpoint '{}', epoch: {}".
      format(eval_model, chkpoint['epoch']))
model.to(device)
model.eval()


def compute_cls(captions_file_prefix, data_type):
    val_sets = {}
    val_sets['all'] = []
    for senti_id, senti in enumerate(sentiment_categories):
        val_sets[senti] = []
        fn = '%s_%s_%s.txt' % (captions_file_prefix, senti, data_type)
        with open(fn, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split()
            line = [int(l) for l in line]
            val_sets[senti].append([senti_id, line])
            val_sets['all'].append([senti_id, line])

    val_datas = {}
    for senti in val_sets:
        val_datas[senti] = get_senti_sents_dataloader(val_sets[senti], idx2word.index('<PAD>'), max_seq_len,
                                                      shuffle=False)

    for senti, val_data in val_datas.items():
        all_num = 0
        wrong_num = 0
        with torch.no_grad():
            for sentis, (caps_tensor, lengths) in tqdm.tqdm(val_data):
                sentis = sentis.to(device)
                caps_tensor = caps_tensor.to(device)

                rest, _ = model.sample(caps_tensor, lengths)
                rest = torch.LongTensor(np.array(rest)).to(device)
                all_num += int(sentis.size(0))
                wrong_num += int((sentis != rest).sum())
        wrong_rate = wrong_num / all_num
        print('%s wrong_rate: %.6f' % (senti, wrong_rate))


if __name__ == "__main__":
    compute_cls(sys.argv[1], sys.argv[2])
