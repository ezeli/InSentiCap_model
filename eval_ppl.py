import json
import sys
import numpy as np
import os
# import kenlm

sentis = ['positive', 'negative', 'neutral']
lm_dir = './data/corpus'
# idx2word = json.load(open('./data/captions/idx2word.json', 'r'))
# word2idx = {}
# for i, w in enumerate(idx2word):
#     word2idx[w] = i
lm_cmd = '~/srilm-1.7.3/bin/i686-m64/ngram -ppl %s -lm ./data/corpus/%s.sri'


# def compute_ppl(captions_file_prefix):
#     lms = {}
#     for senti in sentis:
#         lms[senti] = kenlm.LanguageModel(os.path.join(lm_dir, '%s.arpa' % senti))
#     captions = json.load(open('%s.json' % captions_file_prefix, 'r'))
#     sentiments = json.load(open('%s_sentis.json' % captions_file_prefix, 'r'))
#     all_scores = []
#     for cap in captions:
#         lm = lms[sentiments[cap['image_id']]]
#         caption = cap['caption']
#         caption = caption.split()
#         caption = [str(word2idx[w]) for w in caption] + ['2']
#         caption = ' '.join(caption)
#         full_scores = lm.full_scores(caption, bos=True, eos=True)
#         lp, _, _ = zip(*full_scores)
#         # import pdb
#         # pdb.set_trace()
#         lp /= np.log10(2)
#         lp *= -1
#         mean_score = np.mean(lp)
#         all_scores.append(mean_score)
#     score = np.mean(np.array(all_scores))
#     print('ppl score: ', score)
#     return score
def compute_ppl(captions_file_prefix, data_type):
    lm_cmds = {}
    for senti in sentis:
        lm_cmds[senti] = lm_cmd % ('%s_%s_%s_w.txt' % (captions_file_prefix, senti, data_type), senti)
    # print('lm cms:', lm_cmds)
    scores = {}
    for senti, cmd in lm_cmds.items():
        out = os.popen(cmd).read().split()
        try:
            scores[senti] = float(out[out.index('ppl=') + 1])
        except Exception:
            scores[senti] = 0

    print('ppl scores:', scores)
    print('ppl scores sum:', sum(scores.values()))
    return scores


if __name__ == "__main__":
    compute_ppl(sys.argv[1], sys.argv[2])
