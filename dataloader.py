# coding:utf8
import torch
from torch.utils import data
import numpy as np
import h5py
import random


def create_collate_fn(name, pad_index=0, max_seq_len=17, num_concepts=10,
                      num_sentiments=5):
    def caption_collate_fn(dataset):
        tmp = []
        for fn, fc_feat, att_feat, caps_idx, cpts_idx in dataset:
            for cap in caps_idx:
                tmp.append([fn, fc_feat, att_feat, cap, cpts_idx])
        dataset = tmp
        dataset.sort(key=lambda p: len(p[3]), reverse=True)
        fns, fc_feats, att_feats, caps, cpts = zip(*dataset)
        fc_feats = torch.FloatTensor(np.array(fc_feats))
        att_feats = torch.FloatTensor(np.array(att_feats))

        lengths = [min(len(c), max_seq_len) for c in caps]
        caps_tensor = torch.LongTensor(len(caps), lengths[0]).fill_(pad_index)
        for i, c in enumerate(caps):
            end = lengths[i]
            caps_tensor[i, :end] = torch.LongTensor(c[:end])
        lengths = [l-1 for l in lengths]

        cpts_tensor = torch.LongTensor(len(cpts), num_concepts).fill_(pad_index)
        for i, c in enumerate(cpts):
            end = min(len(c), num_concepts)
            cpts_tensor[i, :end] = torch.LongTensor(c[:end])

        return fns, fc_feats, att_feats, (caps_tensor, lengths), cpts_tensor

    def rl_fact_collate_fn(dataset):
        ground_truth = {}
        tmp = []
        for fn, caps_idx, fc_feat, att_feat, cpts_idx, sentis_idx in dataset:
            ground_truth[fn] = [c[:max_seq_len] for c in caps_idx]
            for cap in caps_idx:
                tmp.append([fn, cap, fc_feat, att_feat, cpts_idx, sentis_idx])
            # cap = random.sample(caps_idx, 1)[0]
            # tmp.append([fn, cap, fc_feat, att_feat, cpts_idx, sentis_idx])
        dataset = tmp
        dataset.sort(key=lambda p: len(p[1]), reverse=True)

        fns, caps, fc_feats, att_feats, cpts, sentis = zip(*dataset)
        fc_feats = torch.FloatTensor(np.array(fc_feats))
        att_feats = torch.FloatTensor(np.array(att_feats))

        lengths = [min(len(c), max_seq_len) for c in caps]
        caps_tensor = torch.LongTensor(len(caps), lengths[0]).fill_(pad_index)
        for i, c in enumerate(caps):
            end = lengths[i]
            caps_tensor[i, :end] = torch.LongTensor(c[:end])
        lengths = [l - 1 for l in lengths]

        cpts_tensor = torch.LongTensor(len(cpts), num_concepts).fill_(pad_index)
        for i, c in enumerate(cpts):
            end = min(len(c), num_concepts)
            cpts_tensor[i, :end] = torch.LongTensor(c[:end])

        sentis_tensor = torch.LongTensor(len(sentis), num_sentiments).fill_(pad_index)
        for i, s in enumerate(sentis):
            end = min(len(s), num_sentiments)
            sentis_tensor[i, :end] = torch.LongTensor(s[:end])

        return fns, fc_feats, att_feats, (caps_tensor, lengths), cpts_tensor, sentis_tensor, ground_truth

    def rl_senti_collate_fn(dataset):
        fns, fc_feats, att_feats, cpts, sentis, senti_labels = zip(*dataset)
        fc_feats = torch.FloatTensor(np.array(fc_feats))
        att_feats = torch.FloatTensor(np.array(att_feats))
        senti_labels = torch.LongTensor(np.array(senti_labels))

        cpts_tensor = torch.LongTensor(len(cpts), num_concepts).fill_(pad_index)
        for i, c in enumerate(cpts):
            end = min(len(c), num_concepts)
            cpts_tensor[i, :end] = torch.LongTensor(c[:end])

        sentis_tensor = torch.LongTensor(len(sentis), num_sentiments).fill_(pad_index)
        for i, s in enumerate(sentis):
            end = min(len(s), num_sentiments)
            sentis_tensor[i, :end] = torch.LongTensor(s[:end])

        return fns, fc_feats, att_feats, cpts_tensor, sentis_tensor, senti_labels

    def iter_fact_collate_fn(dataset):
        tmp = []
        for fn, caps_idx, fc_feat, att_feat, cpts_idx, sentis_idx in dataset:
            for cap in caps_idx:
                tmp.append([fn, cap, fc_feat, att_feat, cpts_idx, sentis_idx])
        dataset = tmp
        dataset.sort(key=lambda p: len(p[1]), reverse=True)

        fns, caps, fc_feats, att_feats, cpts, sentis = zip(*dataset)
        fc_feats = torch.FloatTensor(np.array(fc_feats))
        att_feats = torch.FloatTensor(np.array(att_feats))

        lengths = [min(len(c), max_seq_len) for c in caps]
        caps_tensor = torch.LongTensor(len(caps), lengths[0]).fill_(pad_index)
        for i, c in enumerate(caps):
            end = lengths[i]
            caps_tensor[i, :end] = torch.LongTensor(c[:end])
        lengths = [l-1 for l in lengths]

        cpts_tensor = torch.LongTensor(len(cpts), num_concepts).fill_(pad_index)
        for i, c in enumerate(cpts):
            end = min(len(c), num_concepts)
            cpts_tensor[i, :end] = torch.LongTensor(c[:end])

        sentis_tensor = torch.LongTensor(len(sentis), num_sentiments).fill_(pad_index)
        for i, s in enumerate(sentis):
            end = min(len(s), num_sentiments)
            sentis_tensor[i, :end] = torch.LongTensor(s[:end])

        return fns, fc_feats, att_feats, (caps_tensor, lengths), cpts_tensor, sentis_tensor

    def iter_senti_collate_fn(dataset):
        fns, fc_feats, att_feats, cpts, sentis, senti_labels = zip(*dataset)
        fc_feats = torch.FloatTensor(np.array(fc_feats))
        att_feats = torch.FloatTensor(np.array(att_feats))
        senti_labels = torch.LongTensor(np.array(senti_labels))

        cpts_tensor = torch.LongTensor(len(cpts), num_concepts).fill_(pad_index)
        for i, c in enumerate(cpts):
            end = min(len(c), num_concepts)
            cpts_tensor[i, :end] = torch.LongTensor(c[:end])

        sentis_tensor = torch.LongTensor(len(sentis), num_sentiments).fill_(pad_index)
        for i, s in enumerate(sentis):
            end = min(len(s), num_sentiments)
            sentis_tensor[i, :end] = torch.LongTensor(s[:end])

        return fns, fc_feats, att_feats, cpts_tensor, sentis_tensor, senti_labels

    def concept_collate_fn(dataset):
        fns, fc_feats, cpts = zip(*dataset)
        fc_feats = torch.FloatTensor(np.array(fc_feats))
        cpts_tensors = torch.LongTensor(np.array(cpts))
        return fns, fc_feats, cpts_tensors

    def senti_image_collate_fn(dataset):
        fns, att_feats, labels = zip(*dataset)
        att_feats = torch.FloatTensor(np.array(att_feats))
        labels = torch.LongTensor(np.array(labels))
        return fns, att_feats, labels

    def senti_sents_collate_fn(dataset):
        dataset.sort(key=lambda p: len(p[1]), reverse=True)
        sentis, caps = zip(*dataset)
        sentis = torch.LongTensor(np.array(sentis))

        lengths = [min(len(c), max_seq_len) for c in caps]
        caps_tensor = torch.LongTensor(len(caps), lengths[0]).fill_(pad_index)
        for i, c in enumerate(caps):
            end = lengths[i]
            caps_tensor[i, :end] = torch.LongTensor(c[:end])

        return sentis, (caps_tensor, lengths)

    if name == 'caption':
        return caption_collate_fn
    elif name == 'concept':
        return concept_collate_fn
    elif name == 'senti_image':
        return senti_image_collate_fn
    elif name == 'iter_fact':
        return iter_fact_collate_fn
    elif name == 'iter_senti':
        return iter_senti_collate_fn
    elif name == 'rl_fact':
        return rl_fact_collate_fn
    elif name == 'rl_senti':
        return rl_senti_collate_fn
    elif name == 'senti_sents':
        return senti_sents_collate_fn


class CaptionDataset(data.Dataset):
    def __init__(self, fc_feats, att_feats, img_captions, img_det_concepts):
        self.fc_feats = fc_feats
        self.att_feats = att_feats
        self.captions = list(img_captions.items())  # [(fn, [[1, 2],[3, 4],...]),...]
        self.det_concepts = img_det_concepts  # {fn: [1,2,...])}

    def __getitem__(self, index):
        fn, caps = self.captions[index]
        f_fc = h5py.File(self.fc_feats, mode='r')
        f_att = h5py.File(self.att_feats, mode='r')
        fc_feat = f_fc[fn][:]
        att_feat = f_att[fn][:]
        cpts = self.det_concepts[fn]
        return fn, np.array(fc_feat), np.array(att_feat), caps, cpts

    def __len__(self):
        return len(self.captions)


class RLFactDataset(data.Dataset):
    def __init__(self, fc_feats, att_feats, img_captions,
                 img_det_concepts, img_det_sentiments):
        self.fc_feats = fc_feats
        self.att_feats = att_feats
        self.captions = list(img_captions.items())
        self.det_concepts = img_det_concepts  # {fn: [1,2,...])}
        self.det_sentiments = img_det_sentiments  # {fn: [5,10,...])}

    def __getitem__(self, index):
        fn, caps = self.captions[index]
        f_fc = h5py.File(self.fc_feats, mode='r')
        f_att = h5py.File(self.att_feats, mode='r')
        fc_feat = f_fc[fn][:]
        att_feat = f_att[fn][:]
        cpts = self.det_concepts[fn]
        sentis = self.det_sentiments[fn]
        # cpts_idx = [self.word2idx[w] for w in cpts]
        # sentis_idx = [self.word2idx[w] for w in sentis]
        return fn, caps, np.array(fc_feat), np.array(att_feat), cpts, sentis

    def __len__(self):
        return len(self.captions)


class RLSentiDataset(data.Dataset):
    def __init__(self, fc_feats, att_feats, img_det_concepts,
                 img_det_sentiments, img_senti_labels):
        self.fc_feats = fc_feats
        self.att_feats = att_feats
        self.det_concepts = img_det_concepts  # {fn: [1,2,...])}
        self.det_sentiments = img_det_sentiments  # {fn: [5,10,...])}
        self.img_senti_labels = img_senti_labels  # [(fn, senti_label),...]

    def __getitem__(self, index):
        fn, senti_label = self.img_senti_labels[index]
        f_fc = h5py.File(self.fc_feats, mode='r')
        f_att = h5py.File(self.att_feats, mode='r')
        fc_feat = f_fc[fn][:]
        att_feat = f_att[fn][:]
        cpts = self.det_concepts[fn]
        sentis = self.det_sentiments[fn]
        return fn, np.array(fc_feat), np.array(att_feat), cpts, sentis, senti_label

    def __len__(self):
        return len(self.img_senti_labels)


class IterFactDataset(data.Dataset):
    def __init__(self, fc_feats, att_feats, img_captions, img_det_concepts,
                 img_det_sentiments):
        self.fc_feats = fc_feats
        self.att_feats = att_feats
        self.captions = list(img_captions.items())
        self.det_concepts = img_det_concepts  # {fn: ['a','b',...])}
        self.det_sentiments = img_det_sentiments  # {fn: ['a','b',...])}

    def __getitem__(self, index):
        fn, caps = self.captions[index]
        f_fc = h5py.File(self.fc_feats, mode='r')
        f_att = h5py.File(self.att_feats, mode='r')
        fc_feat = f_fc[fn][:]
        att_feat = f_att[fn][:]
        cpts = self.det_concepts[fn]
        sentis = self.det_sentiments[fn]
        return fn, random.sample(caps, 1), np.array(fc_feat), np.array(att_feat), cpts, sentis

    def __len__(self):
        return len(self.captions)


class IterSentiDataset(data.Dataset):
    def __init__(self, fc_feats, att_feats, img_det_concepts,
                 img_det_sentiments, img_senti_labels):
        self.fc_feats = fc_feats
        self.att_feats = att_feats
        self.det_concepts = img_det_concepts  # {fn: ['a','b',...])}
        self.det_sentiments = img_det_sentiments  # {fn: ['a','b',...])}
        self.img_senti_labels = img_senti_labels  # [(fn, senti_label),...]

    def __getitem__(self, index):
        fn, senti_label = self.img_senti_labels[index]
        f_fc = h5py.File(self.fc_feats, mode='r')
        f_att = h5py.File(self.att_feats, mode='r')
        fc_feat = f_fc[fn][:]
        att_feat = f_att[fn][:]
        cpts = self.det_concepts[fn]
        sentis = self.det_sentiments[fn]
        return fn, np.array(fc_feat), np.array(att_feat), cpts, sentis, senti_label

    def __len__(self):
        return len(self.img_senti_labels)


class ConceptDataset(data.Dataset):
    def __init__(self, fc_feats, img_concepts, idx2concept):
        self.fc_feats = fc_feats
        self.concepts = list(img_concepts.items())
        self.concept2idx = {}
        for i, w in enumerate(idx2concept):
            self.concept2idx[w] = i
        self.num_cpts = len(idx2concept)

    def __getitem__(self, index):
        fn, cpts = self.concepts[index]
        fc_feat = self.fc_feats[fn][:]
        cpts_idx = [self.concept2idx[cpt]
                    for cpt in cpts if cpt in self.concept2idx]
        cpts = np.zeros(self.num_cpts, dtype=np.int16)
        cpts[cpts_idx] = 1
        return fn, np.array(fc_feat), cpts

    def __len__(self):
        return len(self.concepts)


class SentiImageDataset(data.Dataset):
    def __init__(self, senti_att_feats, img_senti_labels, sentiment_categories):
        self.att_feats = senti_att_feats
        self.img_senti_labels = img_senti_labels  # [(fn, senti_label),...]
        self.senti_label2idx = {}
        for i, w in enumerate(sentiment_categories):
            self.senti_label2idx[w] = i

    def __getitem__(self, index):
        fn, senti_label = self.img_senti_labels[index]
        att_feat = self.att_feats[fn][:]
        senti_idx = self.senti_label2idx[senti_label]
        return fn, np.array(att_feat), senti_idx

    def __len__(self):
        return len(self.img_senti_labels)


class SentiSentDataset(data.Dataset):
    def __init__(self, senti_sentences):
        self.senti_sentences = senti_sentences

    def __getitem__(self, index):
        senti, sent = self.senti_sentences[index]
        return senti, np.array(sent)

    def __len__(self):
        return len(self.senti_sentences)


def get_caption_dataloader(fc_feats, att_feats, img_captions, img_det_concepts,
                           pad_index, max_seq_len, num_concepts,
                           batch_size, num_workers=0, shuffle=True):
    dataset = CaptionDataset(fc_feats, att_feats, img_captions, img_det_concepts)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(
                                     'caption', pad_index, max_seq_len + 1,
                                     num_concepts))
    return dataloader


def get_iter_fact_dataloader(fc_feats, att_feats, img_captions, img_det_concepts,
                             img_det_sentiments, pad_index, max_seq_len,
                             num_concepts, num_sentiments,
                             batch_size, num_workers=0, shuffle=True):
    dataset = IterFactDataset(fc_feats, att_feats, img_captions, img_det_concepts,
                              img_det_sentiments)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(
                                     'iter_fact', pad_index=pad_index,
                                     max_seq_len=max_seq_len + 1,
                                     num_concepts=num_concepts,
                                     num_sentiments=num_sentiments))
    return dataloader


def get_iter_senti_dataloader(fc_feats, att_feats, img_det_concepts,
                              img_det_sentiments, img_senti_labels, pad_index,
                              num_concepts, num_sentiments, batch_size,
                              num_workers=0, shuffle=True):
    dataset = IterSentiDataset(fc_feats, att_feats, img_det_concepts,
                               img_det_sentiments, img_senti_labels)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(
                                     'iter_senti', pad_index=pad_index,
                                     num_concepts=num_concepts,
                                     num_sentiments=num_sentiments))
    return dataloader


def get_rl_fact_dataloader(fc_feats, att_feats, img_captions, img_det_concepts,
                           img_det_sentiments, pad_index, max_seq_len, num_concepts,
                           num_sentiments, batch_size, num_workers=0, shuffle=True):
    dataset = RLFactDataset(fc_feats, att_feats, img_captions,
                            img_det_concepts, img_det_sentiments)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size // 5,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(
                                     'rl_fact', pad_index=pad_index,
                                     max_seq_len=max_seq_len + 1,
                                     num_concepts=num_concepts,
                                     num_sentiments=num_sentiments))
    return dataloader


def get_rl_senti_dataloader(fc_feats, att_feats, img_det_concepts,
                            img_det_sentiments, img_senti_labels, pad_index,
                            num_concepts, num_sentiments, batch_size,
                            num_workers=0, shuffle=True):
    dataset = RLSentiDataset(fc_feats, att_feats, img_det_concepts,
                             img_det_sentiments, img_senti_labels)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(
                                     'rl_senti', pad_index=pad_index,
                                     num_concepts=num_concepts,
                                     num_sentiments=num_sentiments))
    return dataloader


def get_concept_dataloader(fc_feats, img_concepts, idx2concept,
                           batch_size, num_workers=0, shuffle=True):
    dataset = ConceptDataset(fc_feats, img_concepts, idx2concept)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn('concept'))
    return dataloader


def get_senti_image_dataloader(senti_att_feats, img_senti_labels,
                               sentiment_categories,
                               batch_size, num_workers=0, shuffle=True):
    dataset = SentiImageDataset(senti_att_feats, img_senti_labels,
                                sentiment_categories)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn('senti_image'))
    return dataloader


def get_senti_sents_dataloader(senti_sentences, pad_index, max_seq_len,
                               batch_size=80, num_workers=2, shuffle=True):
    dataset = SentiSentDataset(senti_sentences)
    dataloader = data.DataLoader(dataset,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 num_workers=num_workers,
                                 collate_fn=create_collate_fn(
                                     'senti_sents', pad_index=pad_index,
                                     max_seq_len=max_seq_len))
    return dataloader
