# coding:utf8
import torch
import h5py
import json
import tqdm

from opts import parse_opt
from models.captioner import Captioner
from dataloader import get_caption_dataloader

opt = parse_opt()
assert opt.eval_model, 'please input eval_model'
assert opt.result_file, 'please input result_file'

f_fc = h5py.File(opt.fc_feats, mode='r')
f_att = h5py.File(opt.att_feats, mode='r')
img_captions = json.load(open(opt.img_captions, 'r'))
img_det_concepts = json.load(open(opt.img_det_concepts, 'r'))

print("====> loading checkpoint '{}'".format(opt.eval_model))
chkpoint = torch.load(opt.eval_model, map_location=lambda s, l: s)
idx2word = chkpoint['idx2word']
model = Captioner(idx2word, chkpoint['settings'])
model.load_state_dict(chkpoint['model'])
print("====> loaded checkpoint '{}', epoch: {}".
      format(opt.eval_model, chkpoint['epoch']))
model.to(opt.device)
model.eval()

test_captions = {}
for fn in img_captions['test']:
    test_captions[fn] = [[]]
test_data = get_caption_dataloader(opt.fc_feats, opt.att_feats, test_captions,
                                   img_det_concepts, idx2word.index('<PAD>'),
                                   opt.max_seq_len, opt.num_concepts, opt.xe_bs,
                                   opt.xe_num_works, shuffle=False)

results = []
for fns, fc_feats, att_feats, _, cpts_tensor in tqdm.tqdm(train_data):
    fc_feats = fc_feats.to(opt.device)
    att_feats = att_feats.to(opt.device)
    cpts_tensor = cpts_tensor.to(opt.device)
    for i, fn in enumerate(fns):
        captions, _ = model.sample(fc_feats[i], att_feats[i], cpts_tensor[i],
                                   beam_size=opt.beam_size,
                                   max_seq_len=opt.max_seq_len)
        results.append({'image_id': fn, 'caption': captions[0]})

json.dump(results, open(opt.result_file, 'w'))
