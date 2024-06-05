import json
import numpy as np
import sys
import tqdm
from flair.nn import Classifier
from flair.data import Sentence

tagger = Classifier.load('ner-large')

dataset = sys.argv[1]

comments = json.load(open(f"../{dataset}/test.json", "r"))

for d in tqdm.tqdm(comments):
    comment = d['comment'].replace('\n', ' ').replace('\r', ' ')
    sentence = Sentence(comment)    
    tagger.predict(sentence)
    if len(sentence.get_labels()) == 0:
        continue
    res = []
    idx = 0
    for l in sentence.get_labels():
        while idx < l.data_point.start_position and idx < len(sentence):
            res.append(sentence[idx].text)
            idx += 1
        res.append(f'[{l.value}]')
        idx = l.data_point.end_position
    d['comment'] = ' '.join(res)

json.dump(comments, open(f'../{dataset}/test_replace.json', 'w'))
