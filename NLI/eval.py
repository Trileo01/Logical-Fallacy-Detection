import pandas as pd
import numpy as np
import sys
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix

dataset = sys.argv[1]
attack = sys.argv[2]

if attack != '0':
    source_path = f'../data/{dataset}/test_{attack}.source'
    label_path = f'../data/{dataset}/test_{attack}.target'
else:
    source_path = f'../data/{dataset}/test.source'
    label_path = f'../data/{dataset}/test.target'

with open(source_path, "r") as f:
    data = f.readlines()

with open(label_path, "r") as f:
    label = f.readlines()

fallacy_set = set([d.split(' </s> ', 1)[0] for d in data])
data_label = []
for i in range(len(label) // len(fallacy_set)):
    ground_truth = [int(int(l) == 2) for l in label[i*len(fallacy_set):(i+1)*len(fallacy_set)]]
    data_label.append(np.argmax(ground_truth))

if attack != '0':
    pred_path = f'../../results/{dataset}_{attack}/prob.txt'
else:
    pred_path = f'../../results/{dataset}/prob.txt'

with open(pred_path, "r") as f:
    pred = f.readlines()

res = []
for i in range(len(pred) // len(fallacy_set)):
    prob = [float(p) for p in pred[i*len(fallacy_set):(i+1)*len(fallacy_set)]]
    res.append(np.argmax(prob))

import json
json.dump([int(x) for x in res], open(f'{dataset}_{attack}.json', 'w'))

cm = confusion_matrix(data_label, res)

f1 = f1_score(data_label, res, average=None)
precision = precision_score(data_label, res, average=None)
recall = recall_score(data_label, res, average=None)
print(cm)
print(precision, recall, f1)
