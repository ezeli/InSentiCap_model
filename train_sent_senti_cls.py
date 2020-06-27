import argparse
import sys
import pdb
import traceback
from bdb import BdbQuit
import json
import random
import pickle
from nltk.classify import NaiveBayesClassifier
from nltk.classify import accuracy


def _extract_feature(text, word2idx):
    feature = {}
    for word in text:
        idx = word2idx.get(word, word2idx['<UNK>'])
        feature[idx] = True
    feature[word2idx['<EOS>']] = True

    return feature


def train():
    senti_corpus = json.load(open(opt.real_captions, 'r'))['senti']
    idx2word = json.load(open(opt.idx2word, 'r'))
    word2idx = {}
    for i, w in enumerate(idx2word):
        word2idx[w] = i
    senti_label2idx = {}
    for i, w in enumerate(opt.sentiment_categories):
        senti_label2idx[w] = i

    train_set = []
    val_set = []
    for senti, sents in senti_corpus.items():
        features = []
        for sent in sents:
            features.append(_extract_feature(sent, word2idx))
        random.shuffle(features)
        senti_id = senti_label2idx[senti]
        for feature in features[1000:]:
            train_set.append([feature, senti_id])
        for feature in features[:1000]:
            val_set.append([feature, senti_id])

    classifier = NaiveBayesClassifier.train(train_set)
    acc = accuracy(classifier, val_set)
    print('accuracy:', acc)
    pickle.dump(classifier, open(opt.sentiment_classifier, 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_captions', type=str, default='./data/captions/real_captions.json')
    parser.add_argument('--idx2word', type=str, default='./data/captions/idx2word.json')
    parser.add_argument('--sentiment_categories', type=list, default=['positive', 'negative', 'neutral'])
    parser.add_argument('--sentiment_classifier', type=str,
                        default='./checkpoint/sentiment/sentence_sentiment_classifier.pkl')

    opt = parser.parse_args()
    try:
        train()
    except BdbQuit:
        sys.exit(1)
    except Exception:
        traceback.print_exc()
        print('')
        pdb.post_mortem()
        sys.exit(1)
