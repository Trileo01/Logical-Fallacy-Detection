import json
import numpy as np
import sys
import tqdm
import random

dataset = sys.argv[1]

comments = json.load(open(f"../{dataset}/test.json", "r"))

for d in tqdm.tqdm(comments):
    comment = d['comment'].replace('\n', ' ').replace('\r', ' ')
    
    tokens = comment.split()
    res = [t for t in tokens if random.random() >= 0.15]
    d['comment'] = ' '.join(res)

json.dump(comments, open(f'../{dataset}/test_delete.json', 'w'))
