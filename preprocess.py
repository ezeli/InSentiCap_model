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
from copy import deepcopy

from models.encoder import Encoder


concept_pos = ['VERB', 'NOUN']


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


def process_caption_datasets():
    for dataset_nm in opt.dataset_names:
        print('===> process %s dataset' % dataset_nm)
        images = json.load(open(os.path.join(opt.caption_datasets_dir, 'dataset_%s.json' % dataset_nm), 'r'))['images']
        img_captions = {'train': {}, 'val': {}, 'test': {}}
        img_captions_pos = {'train': {}, 'val': {}, 'test': {}}
        img_concepts = {'train': {}, 'val': {}, 'test': {}}
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

        json.dump(img_captions, open(os.path.join(opt.captions_dir, dataset_nm, 'img_captions.json'), 'w'))
        json.dump(img_captions_pos, open(os.path.join(opt.captions_dir, dataset_nm, 'img_captions_pos.json'), 'w'))
        json.dump(img_concepts, open(os.path.join(opt.captions_dir, dataset_nm, 'img_concepts.json'), 'w'))


def process_senti_corpus():
    corpus_type = 'part'
    senti_corpus = json.load(open(os.path.join(opt.corpus_dir, corpus_type, 'senti_corpus.json'), 'r'))

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
                tmp_sents.append(nltk.word_tokenize(sent.strip().lower()))
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

    json.dump(tmp_senti_corpus, open(os.path.join(opt.corpus_dir, corpus_type, 'tmp_senti_corpus.json'), 'w'))
    json.dump(tmp_senti_corpus_pos, open(os.path.join(opt.corpus_dir, corpus_type, 'tmp_senti_corpus_pos.json'), 'w'))

    all_sentis = all_sentis.most_common()
    all_sentis = [w for w in all_sentis if w[1] >= 3]
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

    sentiment_words = {}
    for k in sentis_result:
        sentiment_words[k] = list(sentis_result[k].items())
        sentiment_words[k].sort(key=lambda p: p[1], reverse=True)
        sentiment_words[k] = [w[0] for w in sentiment_words[k]]

    common_rm = []
    pos_rm = []
    neg_rm = []
    for i, w in enumerate(sentiment_words['positive']):
        if w in sentiment_words['negative']:
            n_idx = sentiment_words['negative'].index(w)
            if abs(i - n_idx) < 5:
                common_rm.append(w)
            elif i > n_idx:
                pos_rm.append(w)
            else:
                neg_rm.append(w)
    for w in common_rm:
        sentiment_words['positive'].remove(w)
        sentiment_words['negative'].remove(w)
    for w in pos_rm:
        sentiment_words['positive'].remove(w)
    for w in neg_rm:
        sentiment_words['negative'].remove(w)

    tmp_sentiment_words = {}
    for senti in sentiment_words:
        tmp_sentiment_words[senti] = {}
        for w in sentiment_words[senti]:
            tmp_sentiment_words[senti][w] = sentis_result[senti][w]
    sentiment_words = tmp_sentiment_words

    json.dump(sentiment_words, open(os.path.join(opt.corpus_dir, corpus_type, 'sentiment_words.json'), 'w'))

    tmp_sentiment_words = {}
    tmp_sentiment_words.update(sentiment_words['positive'])
    tmp_sentiment_words.update(sentiment_words['negative'])
    sentiment_words = tmp_sentiment_words

    tmp_sentiment_detector = defaultdict(list)
    for noun, senti_words in sentiment_detector.items():
        number = sum([w[1] for w in senti_words])
        for senti_word in senti_words:
            if senti_word[0] in sentiment_words:
                tmp_sentiment_detector[noun].append(
                    (senti_word[0], senti_word[1] / number * sentiment_words[senti_word[0]]))
    sentiment_detector = tmp_sentiment_detector
    tmp_sentiment_detector = {}
    for noun, senti_words in sentiment_detector.items():
        if len(senti_words) <= 50:
            tmp_sentiment_detector[noun] = senti_words

    json.dump(tmp_sentiment_detector, open(os.path.join(opt.corpus_dir, corpus_type, 'sentiment_detector.json'), 'w'))


def build_idx2concept():
    for dataset_nm in opt.dataset_names:
        img_concepts = json.load(open(os.path.join(opt.captions_dir, dataset_nm, 'img_concepts.json'), 'r'))
        tc = Counter()
        for concepts in img_concepts.values():
            for cs in tqdm.tqdm(concepts.values()):
                tc.update(cs)
        tc = tc.most_common()
        idx2concept = [w[0] for w in tc[:2000]]
        json.dump(idx2concept, open(os.path.join(opt.captions_dir, dataset_nm, 'idx2concept.json'), 'w'))


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
    corpus_type = 'part'
    senti_corpus = json.load(open(os.path.join(opt.corpus_dir, corpus_type, 'tmp_senti_corpus.json'), 'r'))
    sentiment_words = json.load(open(os.path.join(opt.corpus_dir, corpus_type, 'sentiment_words.json'), 'r'))
    idx2sentiment = []
    for v in sentiment_words.values():
        idx2sentiment.extend(list(v.keys()))

    for dataset_nm in opt.dataset_names:
        img_captions = json.load(open(os.path.join(opt.captions_dir, dataset_nm, 'img_captions.json'), 'r'))
        idx2concept = json.load(open(os.path.join(opt.captions_dir, dataset_nm, 'idx2concept.json'), 'r'))

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

        idx2word.extend(idx2sentiment)
        idx2word.extend(idx2concept)
        idx2word = list(set(idx2word))
        idx2word = ['<PAD>', '<SOS>', '<EOS>', '<UNK>'] + idx2word
        json.dump(idx2word, open(os.path.join(opt.captions_dir, dataset_nm, corpus_type, 'idx2word.json'), 'w'))


def get_img_det_sentiments():
    corpus_type = 'part'
    sentiment_detector = json.load(open(os.path.join(opt.corpus_dir, corpus_type, 'sentiment_detector.json'), 'r'))

    for dataset_nm in opt.dataset_names:
        det_concepts = json.load(open(os.path.join(opt.captions_dir, dataset_nm, 'img_det_concepts.json'), 'r'))
        det_sentiments = {}
        null_sentis = []
        for fn, concepts in tqdm.tqdm(det_concepts.items()):
            sentis = []
            for con in concepts:
                sentis.extend(sentiment_detector.get(con, []))
            if sentis:
                tmp_sentis = defaultdict(float)
                for w, s in sentis:
                    tmp_sentis[w] += s
                sentis = list(tmp_sentis.items())
                sentis.sort(key=lambda p: p[1], reverse=True)
                sentis = [w[0] for w in sentis]
            else:
                null_sentis.append(fn)
            det_sentiments[fn] = sentis[:20]
        json.dump(det_sentiments, open(os.path.join(opt.captions_dir, dataset_nm, corpus_type, 'img_det_sentiments.json'), 'w'))


def get_senti_captions():
    corpus_type = 'part'
    sentiment_detector = json.load(open(os.path.join(opt.corpus_dir, corpus_type, 'sentiment_detector.json'), 'r'))
    senti_corpus = json.load(open(os.path.join(opt.corpus_dir, corpus_type, 'tmp_senti_corpus.json'), 'r'))
    senti_corpus_pos = json.load(open(os.path.join(opt.corpus_dir, corpus_type, 'tmp_senti_corpus_pos.json'), 'r'))
    sentiment_words = json.load(open(os.path.join(opt.corpus_dir, corpus_type, 'sentiment_words.json'), 'r'))
    idx2sentiment = []
    for v in sentiment_words.values():
        idx2sentiment.extend(list(v.keys()))

    senti_captions = defaultdict(list)  # len(pos) = 4633, len(neg) = 3760
    cpts_len = defaultdict(int)  # len = 23, we choose 5
    sentis_len = defaultdict(int)  # len = 104, we choose 5 or 10
    wrong = []  # len = 476
    for senti in senti_corpus:
        for i, cap in enumerate(senti_corpus[senti]):
            pos = senti_corpus_pos[senti][i]
            cpts = []
            for j, p in enumerate(pos):
                if p in concept_pos:
                    cpts.append(cap[j])
            cpts = list(set(cpts))
            sentis = []
            for con in cpts:
                sentis.extend(sentiment_detector.get(con, []))
            if sentis:
                tmp_sentis = defaultdict(float)
                for w, s in sentis:
                    tmp_sentis[w] += s
                sentis = list(tmp_sentis.items())
                sentis.sort(key=lambda p: p[1], reverse=True)
                sentis = [w[0] for w in sentis]
                senti_captions[senti].append([cap, cpts[:20], sentis[:20]])
                cpts_len[len(cpts)] += 1
                sentis_len[len(sentis)] += 1
            else:
                wrong.append([len(cpts), len(sentis)])
    cpts_len = list(cpts_len.items())
    cpts_len.sort()
    sentis_len = list(sentis_len.items())
    sentis_len.sort()

    for dataset_nm in opt.dataset_names:
        cpts_len = defaultdict(int)  # len = 23, we choose 5
        sentis_len = defaultdict(int)  # len = 104, we choose 5 or 10
        wrong = []
        img_captions = json.load(open(os.path.join(opt.captions_dir, dataset_nm, 'img_captions.json'), 'r'))['train']
        img_captions_pos = json.load(open(os.path.join(opt.captions_dir, dataset_nm, 'img_captions_pos.json'), 'r'))['train']
        fact_caps = []
        for fn, caps in tqdm.tqdm(img_captions.items()):
            for i, cap in enumerate(caps):
                flag = True
                for w in cap:
                    if w in idx2sentiment:
                        flag = False
                        break
                if flag:
                    pos = img_captions_pos[fn][i]
                    cpts = []
                    for j, p in enumerate(pos):
                        if p in concept_pos:
                            cpts.append(cap[j])
                    cpts = list(set(cpts))
                    sentis = []
                    for con in cpts:
                        sentis.extend(sentiment_detector.get(con, []))
                    if sentis:
                        tmp_sentis = defaultdict(float)
                        for w, s in sentis:
                            tmp_sentis[w] += s
                        sentis = list(tmp_sentis.items())
                        sentis.sort(key=lambda p: p[1], reverse=True)
                        sentis = [w[0] for w in sentis]
                        fact_caps.append([cap, cpts[:20], sentis[:20]])
                        cpts_len[len(cpts)] += 1
                        sentis_len[len(sentis)] += 1
                    else:
                        wrong.append([len(cpts), len(sentis)])
        cpts_len = list(cpts_len.items())
        cpts_len.sort()
        sentis_len = list(sentis_len.items())
        sentis_len.sort()

        tmp_senti_captions = deepcopy(senti_captions)
        tmp_senti_captions['neutral'] = fact_caps
        json.dump(tmp_senti_captions, open(os.path.join(opt.captions_dir, dataset_nm, corpus_type, 'senti_captions.json'), 'w'))


def get_anno_captions():
    for dataset_nm in opt.dataset_names:
        images = json.load(open(os.path.join(opt.caption_datasets_dir, 'dataset_%s.json' % dataset_nm), 'r'))['images']
        anno_captions = {}
        for image in tqdm.tqdm(images):
            if image['split'] == 'test':
                fn = image['filename']
                sentences = []
                for sentence in image['sentences']:
                    raw = sentence['raw'].strip().lower()
                    sentences.append(raw)
                anno_captions[fn] = sentences
        json.dump(anno_captions, open(os.path.join(opt.captions_dir, dataset_nm, 'anno_captions.json'), 'w'))


def get_lm_sents():
    corpus_type = 'part'
    for dataset_nm in opt.dataset_names:
        senti_captions = json.load(open(os.path.join(opt.captions_dir, dataset_nm, corpus_type, 'senti_captions.json'), 'r'))
        for senti in senti_captions:
            senti_captions[senti] = [' '.join(c[0]) for c in senti_captions[senti]]
        senti_sents = defaultdict(str)
        for senti in senti_captions:
            for cap in senti_captions[senti]:
                senti_sents[senti] += cap + '\n'

        lm_dir = os.path.join(opt.captions_dir, dataset_nm, corpus_type, 'lm')
        if not os.path.exists(lm_dir):
            os.makedirs(lm_dir)
        for senti in senti_sents:
            with open(os.path.join(lm_dir, '%s_w.txt' % senti), 'w') as f:
                f.write(senti_sents[senti])

    count_cmd = 'ngram-count -text %s -order 3 -write %s'
    lm_cmd = 'ngram-count -read %s -order 3 -lm %s -interpolate -kndiscount'
    for dataset_nm in opt.dataset_names:
        lm_dir = os.path.join(opt.captions_dir, dataset_nm, corpus_type, 'lm')
        fns = os.listdir(lm_dir)
        for fn in fns:
            if fn.endswith('_w.txt'):
                txt_file = os.path.join(lm_dir, fn)
                count_file = os.path.join(lm_dir, '%s.count' % fn.split('.')[0])
                lm_file = os.path.join(lm_dir, '%s.sri' % fn.split('.')[0])
                out = os.popen(count_cmd % (txt_file, count_file)).read()
                print(out)
                out = os.popen(lm_cmd % (count_file, lm_file)).read()
                print(out)

    # for kenlm
    kenlm_cmd = "lmplz -o 3 <%s >%s"
    for dataset_nm in opt.dataset_names:
        senti_captions = json.load(
            open(os.path.join(opt.captions_dir, dataset_nm, corpus_type, 'senti_captions.json'), 'r'))
        for senti in senti_captions:
            senti_captions[senti] = [c[0] for c in senti_captions[senti]]
        idx2word = json.load(open(os.path.join(opt.captions_dir, dataset_nm, corpus_type, 'idx2word.json'), 'r'))
        word2idx = {}
        for i, w in enumerate(idx2word):
            word2idx[w] = i

        senti_captions_id = {}
        for senti in senti_captions:
            senti_captions_id[senti] = []
            for cap in senti_captions[senti]:
                tmp = [word2idx.get(w, None) or word2idx['<UNK>'] for w in cap] + [word2idx['<EOS>']]
                tmp = ' '.join([str(idx) for idx in tmp])
                senti_captions_id[senti].append(tmp)
        lm_dir = os.path.join(opt.captions_dir, dataset_nm, corpus_type, 'lm')
        for senti in senti_captions_id:
            senti_captions_id[senti] = '\n'.join(senti_captions_id[senti])
            with open(os.path.join(lm_dir, '%s_id.txt' % senti), 'w') as f:
                f.write(senti_captions_id[senti])
            out = os.popen(kenlm_cmd % (os.path.join(lm_dir, '%s_id.txt' % senti), os.path.join(lm_dir, '%s_id.kenlm.arpa' % senti))).read()
            print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--imgs_dir', type=str, default='./data/images/sentiment')
    parser.add_argument('--feats_dir', type=str, default='./data/features/sentiment')
    parser.add_argument('--resnet101_file', type=str,
                        default='./data/pre_models/resnet101.pth')

    parser.add_argument('--caption_datasets_dir', type=str, default='../../dataset/caption/caption_datasets')
    parser.add_argument('--dataset_names', type=list, default=['flickr30k', 'coco'])
    parser.add_argument('--captions_dir', type=str, default='./data/captions/')

    parser.add_argument('--corpus_dir', type=str, default='./data/corpus')

    parser.add_argument('--senti_imgs_dir', type=str, default='./data/images/sentiment')
    parser.add_argument('--img_senti_labels', type=str, default='./data/captions/img_senti_labels.json')

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
