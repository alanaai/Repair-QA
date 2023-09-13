import sys
import json

path = sys.argv[1]
pred_file = json.load(open(path, 'r'))

hyps, refs = [], []

null_pred = 0
for d in pred_file:
    if d['hypothesis'] != '':
        hyps.append(d['hypothesis'])
        refs.append(d['reference'])
    else:
        null_pred += 1

if null_pred >0:
    print(f"Null hypothesis predictions: {null_pred}")

assert len(refs) == len(hyps)

# compute ROUGE
from rouge import Rouge
rouge = Rouge()
scores = rouge.get_scores(hyps, refs, avg=True)
print(scores)


# compute BLEU-4 score
from nltk.translate.bleu_score import corpus_bleu
refs = [[x.split()] for x in refs]
hyps = [x.split() for x in hyps]
bleu = corpus_bleu(refs, hyps)
print(f'BLEU-4: {bleu*100}')