import sys
import pickle
from nltk.classify import accuracy

sentis = ['positive', 'negative', 'neutral']
classifier = pickle.load(open('./checkpoint/sentiment/sentence_sentiment_classifier_w.pkl', 'rb'))


def _extract_feature(text):
    text = text.strip().split()
    feature = {}
    for word in text:
        feature[word] = True

    return feature


def compute_cls(captions_file_prefix, data_type):
    val_set = []
    for senti in sentis:
        tmp_set = []
        fn = '%s_%s_%s_w.txt' % (captions_file_prefix, senti, data_type)
        with open(fn, 'r') as f:
            lines = f.readlines()
        for line in lines:
            tmp_set.append([_extract_feature(line), senti])
        val_set.extend(tmp_set)
        acc = accuracy(classifier, tmp_set)
        print('%s accuracy: %s' % (senti, acc))

    acc = accuracy(classifier, val_set)
    print('all accuracy: %s' % acc)


if __name__ == "__main__":
    compute_cls(sys.argv[1], sys.argv[2])
