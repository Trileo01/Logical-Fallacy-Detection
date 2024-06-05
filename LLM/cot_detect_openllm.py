import json
import numpy as np
import sys
import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import ollama

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)

dataset = sys.argv[1]
attack = sys.argv[2]
model = sys.argv[3]

client = ollama

if attack != '0':
    comments = json.load(open(f"../{dataset}/test_{attack}.json", "r"))
else:
    comments = json.load(open(f"../{dataset}/test.json", "r"))

prediction = []
label = []
for d in tqdm.tqdm(comments):
    comment = d['comment'].replace('\n', ' ').replace('\r', ' ')
    parent = d['parent'].replace('\n', ' ').replace('\r', ' ') if 'parent' in d else 'none'
    if 'attack' in d:
        comment += ' ' + d['attack'].replace('\n', ' ').replace('\r', ' ')

    messages = [
            {"role": "user", "content": f"Determine the presence of a logical fallacy in the given COMMENT through the logic and reasoning of the content. If the available information is insufficient for detection, output \"unknown.\" Utilize the TITLE and PARENT_COMMENT as context to support your decision, and provide an explanation of the reasoning behind your determination.\n\nLet's think step by step. First, answer these questions:\nWhat are the key indicators of a logical fallacy?\nHow is reasoning affected by a logical fallacy?\nIn sentences with logical fallacies, are there any common patterns?\nHow does the context of a sentence affect the presence of a logical fallacy?\n\n Then, use the answers to these questions to determine the presence of a logical fallacy in the given COMMENT. The output format should be [YES/NO/UNKNOWN] [EXPLANATIONS]\nTITLE: {d['title'] if 'title' in d else 'none'}\nPARENT_COMMENT: {parent if 'parent' in d else 'none'}  \nCOMMENT:{comment}"}
    ]

    response = client.chat(
        model=model,
        messages=messages,
        stream=False
    )
    response = response['message']['content'].lower()
    pred = response.replace('[', ' ').replace(']', ' ').replace('"', ' ').replace("*", ' ').split()
    yes = 0
    no = 0
    for t in pred:
        if t == 'yes':
            yes += 1
        if t == 'no':
            no += 1
    prediction.append(1 if yes >= no else 0)
    label.append(0 if d['fallacy'] == "none" else 1)


json.dump(prediction, open(f'{dataset}_{model}_cot_pred_{attack}.json', 'w'))

cm = confusion_matrix(label, prediction)
precision = precision_score(label, prediction, average=None)
recall = recall_score(label, prediction, average=None)
f1 = f1_score(label, prediction, average=None)

print(dataset)
print(cm)
print(precision, recall, f1)

