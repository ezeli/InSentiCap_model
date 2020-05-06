import argparse
import json
import os
import random
import sys
import pdb
import traceback
from bdb import BdbQuit
import h5py
import tqdm
from collections import Counter, defaultdict
import skimage.io
import nltk
import torch

from models.encoder import Encoder


def extract_imgs_feat():
    encoder = Encoder(opt.resnet101_file)
    encoder.to(opt.device)
    encoder.eval()

    imgs = os.listdir(opt.imgs_dir)
    imgs.sort()
    imgs = imgs[4220:]

    if not os.path.exists(opt.feats_dir):
        os.makedirs(opt.feats_dir)
    with h5py.File(os.path.join(opt.feats_dir, 'feats_fc.h5')) as file_fc, \
            h5py.File(os.path.join(opt.feats_dir, 'feats_att.h5')) as file_att:
        try:
            for img_nm in tqdm.tqdm(imgs):
                img = skimage.io.imread(os.path.join(opt.imgs_dir, img_nm))
                if len(img.shape) == 3 and img.shape[-1] == 4:
                    img = img[:, :, :3]
                with torch.no_grad():
                    img = encoder.preprocess(img)
                    img = img.to(opt.device)
                    img_fc, img_att = encoder(img)
                file_fc.create_dataset(img_nm, data=img_fc.cpu().float().numpy())
                file_att.create_dataset(img_nm, data=img_att.cpu().float().numpy())
        except BaseException as e:
            file_fc.close()
            file_att.close()
            print('--------------------------------------------------------------------')
            raise e


def process_coco_captions():
    images = json.load(open(opt.dataset_coco, 'r'))['images']
    img_captions = {'train': {}, 'val': {}, 'test': {}}
    img_captions_pos = {'train': {}, 'val': {}, 'test': {}}
    img_concepts = {'train': {}, 'val': {}, 'test': {}}
    concept_pos = ['VERB', 'NOUN']
    for image in tqdm.tqdm(images):
        fn = image['filename']
        split = image['split']
        if split == 'restval':
            split = 'train'
        img_captions[split][fn] = []
        img_captions_pos[split][fn] = []
        img_concepts[split][fn] = set()
        sentences = []
        for sentence in image['sentences']:
            raw = sentence['raw'].lower()
            words = nltk.word_tokenize(raw)
            sentences.append(words)
        tagged_sents = nltk.pos_tag_sents(sentences, tagset='universal')
        for tagged_tokens in tagged_sents:
            words = []
            poses = []
            for w, p in tagged_tokens:
                if p == '.':  # remove punctuation
                    continue
                words.append(w)
                poses.append(p)
                if p in concept_pos:
                    img_concepts[split][fn].add(w)
            img_captions[split][fn].append(words)
            img_captions_pos[split][fn].append(poses)
        img_concepts[split][fn] = list(img_concepts[split][fn])

    json.dump(img_captions, open(opt.img_captions, 'w'))
    json.dump(img_captions_pos, open(opt.img_captions_pos, 'w'))
    json.dump(img_concepts, open(opt.img_concepts, 'w'))

    # lens = {}
    # for split in img_captions:
    #     for fn in tqdm.tqdm(img_captions[split]):
    #         for i in range(len(img_captions[split][fn])):
    #             if len(img_captions[split][fn][i]) != len(img_captions_pos[split][fn][i]):
    #                 print(split, fn)
    #                 raise Exception('xxx')
    #         l = len(img_concepts[split][fn])
    #         lens[l] = lens.get(l, 0) + 1
    # # lens: [(3, 1), (4, 2), (5, 13), (6, 116), (7, 387), (8, 1136), (9, 2579), (10, 4863), (11, 7494), (1
    # # 2, 10532), (13, 13086), (14, 14592), (15, 14782), (16, 13659), (17, 11558), (18, 9420), (19, 6
    # # 901), (20, 4671), (21, 3050), (22, 1790), (23, 1134), (24, 656), (25, 370), (26, 200), (27, 12
    # # 1), (28, 61), (29, 52), (30, 20), (31, 18), (32, 5), (33, 9), (34, 1), (35, 3), (36, 3), (38,
    # # 1), (41, 1)]


def process_senti_corpus():
    senti_corpus = json.load(open(opt.senti_corpus, 'r'))

    tmp_senti_corpus = defaultdict(list)
    tmp_senti_corpus_pos = defaultdict(list)
    all_sentis = Counter()
    sentis = defaultdict(Counter)
    sentiment_detector = defaultdict(Counter)

    for senti_label, sents in senti_corpus.items():
        for i in tqdm.tqdm(range(0, len(sents), 100)):
            cur_sents = sents[i:i + 100]
            tmp_sents = []
            for sent in cur_sents:
                tmp_sents.append(nltk.word_tokenize(sent.lower()))
            tagged_sents = nltk.pos_tag_sents(tmp_sents, tagset='universal')
            for tagged_tokens in tagged_sents:
                words = []
                poses = []
                nouns = []
                adjs = []
                for w, p in tagged_tokens:
                    if p == '.':  # remove punctuation
                        continue
                    words.append(w)
                    poses.append(p)
                    if p == 'ADJ':
                        adjs.append(w)
                    elif p == 'NOUN':
                        nouns.append(w)
                tmp_senti_corpus[senti_label].append(words)
                tmp_senti_corpus_pos[senti_label].append(poses)
                if adjs:
                    all_sentis.update(adjs)
                    sentis[senti_label].update(adjs)
                    for noun in nouns:
                        sentiment_detector[noun].update(adjs)

    # json.dump(tmp_senti_corpus, open(opt.tmp_senti_corpus, 'w'))
    # json.dump(tmp_senti_corpus_pos, open(opt.tmp_senti_corpus_pos, 'w'))

    all_sentis = all_sentis.most_common()[:1000]
    sentis = {k: v.most_common() for k, v in sentis.items()}
    sentiment_detector = {k: v.most_common() for k, v in sentiment_detector.items()}

    all_sentis = {k: v for k, v in all_sentis}

    len_sentis = defaultdict(int)
    for k, v in sentis.items():
        for _, n in v:
            len_sentis[k] += n
    tf_sentis = defaultdict(dict)
    tmp_sentis = defaultdict(dict)
    for k, v in sentis.items():
        for w, n in v:
            tf_sentis[k][w] = n / len_sentis[k]
            tmp_sentis[k][w] = n
    sentis = tmp_sentis

    sentis_result = defaultdict(dict)
    for k, v in tf_sentis.items():
        for w, tf in v.items():
            if w in all_sentis:
                sentis_result[k][w] = tf * (sentis[k][w] / all_sentis[w])

    pos_words = list(sentis_result['positive'].items())
    pos_words.sort(key=lambda p: p[1], reverse=True)
    neg_words = list(sentis_result['negative'].items())
    neg_words.sort(key=lambda p: p[1], reverse=True)
    # neu_words = list(sentis_result['neutral'].items())
    # neu_words.sort(key=lambda p: p[1], reverse=True)
    sentiment_words = {}
    for k, v in pos_words[:250]:
        sentiment_words[k] = v
    for k, v in neg_words[:250]:
        if k not in sentiment_words:
            sentiment_words[k] = v
        else:
            sentiment_words[k] = v if v > sentiment_words[k] else sentiment_words[k]

    for k in sentis_result:
        sentis_result[k] = list(sentis_result[k].items())
        sentis_result[k].sort(key=lambda p: p[1], reverse=True)
        sentis_result[k] = [w[0] for w in sentis_result[k][:250]]
    json.dump(sentis_result, open(opt.sentiment_words, 'w'))

    tmp_sentiment_detector = defaultdict(list)
    for noun, senti_words in sentiment_detector.items():
        number = sum([w[1] for w in senti_words])
        for senti_word in senti_words:
            if senti_word[0] in sentiment_words:
                tmp_sentiment_detector[noun].append(
                    (senti_word[0], senti_word[1] / number * sentiment_words[senti_word[0]]))
    json.dump(tmp_sentiment_detector, open(opt.sentiment_detector, 'w'))


def build_idx2concept():
    img_concepts = json.load(open(opt.img_concepts, 'r'))
    tc = Counter()
    for concepts in img_concepts.values():
        for cs in tqdm.tqdm(concepts.values()):
            tc.update(cs)
    tc = tc.most_common()
    idx2concept = [w[0] for w in tc[:2000]]
    json.dump(idx2concept, open(opt.idx2concept, 'w'))


def get_img_senti_labels():
    senti_img_fns = os.listdir(opt.senti_imgs_dir)
    senti_imgs = defaultdict(list)
    for fn in senti_img_fns:
        senti = fn.split('_')[0]
        senti_imgs[senti].append((fn, senti))
    random.shuffle(senti_imgs['positive'])
    random.shuffle(senti_imgs['negative'])
    random.shuffle(senti_imgs['neutral'])
    img_senti_labels = {'train': [], 'val': [], 'test': []}
    img_senti_labels['val'].extend(senti_imgs['positive'][:100])
    img_senti_labels['val'].extend(senti_imgs['negative'][:100])
    img_senti_labels['val'].extend(senti_imgs['neutral'][:50])
    img_senti_labels['test'].extend(senti_imgs['positive'][100:200])
    img_senti_labels['test'].extend(senti_imgs['negative'][100:200])
    img_senti_labels['test'].extend(senti_imgs['neutral'][50:100])
    img_senti_labels['train'].extend(senti_imgs['positive'][200:])
    img_senti_labels['train'].extend(senti_imgs['negative'][200:])
    img_senti_labels['train'].extend(senti_imgs['neutral'][100:])
    json.dump(img_senti_labels, open(opt.img_senti_labels, 'w'))


def build_idx2word():
    img_captions = json.load(open(opt.img_captions, 'r'))
    senti_corpus = json.load(open(opt.tmp_senti_corpus, 'r'))
    sentiment_detector = json.load(open(opt.sentiment_detector, 'r'))
    idx2concept = json.load(open(opt.idx2concept, 'r'))

    tc = Counter()
    for captions in img_captions.values():
        for caps in captions.values():
            for cap in caps:
                tc.update(cap)
    for captions in senti_corpus.values():
        for cap in captions:
            tc.update(cap)
    tc = tc.most_common()
    idx2word = [w[0] for w in tc if w[1] > 5]

    senti_words = Counter()
    for sentis in sentiment_detector.values():
        senti_words.update([w[0] for w in sentis])
    senti_words = senti_words.most_common()
    senti_words = [w[0] for w in senti_words]

    idx2word.extend(senti_words)
    idx2word.extend(idx2concept)
    idx2word = list(set(idx2word))
    idx2word = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + idx2word
    json.dump(idx2word, open(opt.idx2word, 'w'))


def get_img_det_sentiments():
    sentiment_detector = json.load(open(opt.sentiment_detector, 'r'))
    det_concepts = json.load(open('./data/captions/img_det_concepts.json', 'r'))
    det_sentiments = {}
    null_sentis = []
    for fn, concepts in tqdm.tqdm(det_concepts.items()):
        sentis = []
        for con in concepts:
            sentis.extend(sentiment_detector.get(con, []))
        if sentis:
            sentis.sort(key=lambda p: p[1], reverse=True)
            sentis = [w[0] for w in sentis]
            sentis = sorted(set(sentis), key=sentis.index)
        else:
            null_sentis.append(fn)
        det_sentiments[fn] = sentis
    json.dump(det_sentiments, open(opt.img_det_sentiments, 'w'))


def get_real_captions():
    senti_corpus = json.load(open(opt.tmp_senti_corpus, 'r'))
    img_captions = json.load(open(opt.img_captions, 'r'))['train']
    sentiment_words = json.load(open(opt.sentiment_words, 'r'))
    real_captions = {'fact': [], 'senti': []}

    caps = []
    for v in img_captions.values():
        caps.extend(v)

    senti_caps = defaultdict(list)
    guiyi = 0
    for cap in tqdm.tqdm(caps):
        num_pos = 0
        num_neg = 0
        for w in cap:
            if w in sentiment_words['positive']:
                num_pos += 1
            if w in sentiment_words['negative']:
                num_neg += 1
        if num_pos == 0 and num_neg == 0:
            senti_caps['neu'].append(cap)
        elif num_pos > 1 and num_neg > 1:
            guiyi += 1
            continue
        elif num_pos > 1:
            senti_caps['pos'].append(cap)
        elif num_neg > 1:
            senti_caps['neg'].append(cap)
    for k, v in senti_caps.items():
        senti_caps[k] = [c for c in v if 3 < len(c) < 25]

    neu_caps = random.sample(senti_caps['neu'], 10000)
    senti_corpus['neutral'].extend(neu_caps)

    fact_caps = senti_caps['neg'] + random.sample(senti_caps['pos'], 30000)
    fact_caps.extend(random.sample(senti_caps['neu'], 100000-len(fact_caps)))
    real_captions['fact'] = fact_caps
    real_captions['senti'] = senti_corpus
    json.dump(real_captions, open(opt.real_captions, 'w'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--imgs_dir', type=str, default='./data/images/sentiment')
    parser.add_argument('--feats_dir', type=str, default='./data/features/sentiment')
    parser.add_argument('--resnet101_file', type=str,
                        default='./data/pre_models/resnet101.pth')

    parser.add_argument('--dataset_coco', type=str, default='../../dataset/caption/coco/dataset_coco.json')
    parser.add_argument('--img_captions', type=str, default='./data/captions/img_captions.json')
    parser.add_argument('--img_captions_pos', type=str, default='./data/captions/img_captions_pos.json')
    parser.add_argument('--img_concepts', type=str, default='./data/captions/img_concepts.json')

    parser.add_argument('--senti_corpus', type=str, default='../../dataset/sentiment/corpus/raws/senti_corpus.json')
    parser.add_argument('--tmp_senti_corpus', type=str, default='./data/captions/tmp_senti_corpus.json')
    parser.add_argument('--tmp_senti_corpus_pos', type=str, default='./data/captions/tmp_senti_corpus_pos.json')
    parser.add_argument('--sentiment_words', type=str, default='./data/captions/tmp_sentiment_words.json')
    parser.add_argument('--sentiment_detector', type=str, default='./data/captions/sentiment_detector.json')

    parser.add_argument('--idx2concept', type=str, default='./data/captions/idx2concept.json')

    parser.add_argument('--senti_imgs_dir', type=str, default='./data/images/sentiment')
    parser.add_argument('--img_senti_labels', type=str, default='./data/captions/img_senti_labels.json')

    parser.add_argument('--idx2word', type=str, default='./data/captions/idx2word.json')

    parser.add_argument('--img_det_sentiments', type=str, default='./data/captions/img_det_sentiments.json')

    parser.add_argument('--real_captions', type=str, default='./data/captions/real_captions.json')

    opt = parser.parse_args()

    opt.use_gpu = torch.cuda.is_available()
    opt.device = torch.device('cuda:0') if opt.use_gpu else torch.device('cpu')

    try:
        # extract_imgs_feat()
        # process_coco_captions()
        process_senti_corpus()
        # build_idx2concept()
        # get_img_senti_labels()
        # build_idx2word()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
