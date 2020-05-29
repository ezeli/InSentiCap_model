import json
import sys
import numpy as np
import os
import kenlm

sentis = ['positive', 'negative', 'neutral']
lm_dir = './data/corpus'
idx2word = json.load(open('./data/captions/idx2word.json', 'r'))
word2idx = {}
for i, w in enumerate(idx2word):
    word2idx[w] = i


def compute_ppl(captions_file_prefix):
    lms = {}
    for senti in sentis:
        lms[senti] = kenlm.LanguageModel(os.path.join(lm_dir, '%s.arpa' % senti))
    captions = json.load(open('%s.json' % captions_file_prefix, 'r'))
    sentiments = json.load(open('%s_sentis.json' % captions_file_prefix, 'r'))
    all_scores = []
    for cap in captions:
        lm = lms[sentiments[cap['image_id']]]
        caption = cap['caption']
        caption = caption.split()
        caption = [str(word2idx[w]) for w in caption]
        caption = ' '.join(caption)
        full_scores = lm.full_scores(caption, bos=True, eos=True)
        lp, _, _ = zip(*full_scores)
        # import pdb
        # pdb.set_trace()
        lp /= np.log10(2)
        lp *= -1
        mean_score = np.mean(lp)
        all_scores.append(mean_score)
    score = np.mean(np.array(all_scores))
    print('ppl score: ', score)
    return score


if __name__ == "__main__":
    compute_ppl(sys.argv[1])
