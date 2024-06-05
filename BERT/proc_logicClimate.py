import json
import sys
from nltk.tokenize import word_tokenize
import pandas as pd

for split in ['train', 'dev', 'test']:
    data = pd.read_csv(f'../../NLI/data/climate_{split}_mh.csv')

    output = open(f"{split}.txt", "w")

    for index, row in data.iterrows():
        if pd.isna(row['source_article']):
            continue

        comment = row['source_article'].replace('\n', ' ').replace('\r', ' ')
        token = word_tokenize(comment)
        token = [f"\"{t}\"" for t in token]
        fallacy = "none"
        binary = "fallacy"
        
        binary = [f"\'{binary}\'"] * len(token)
        multi = [f"\'{fallacy}\'"] * len(token)

        parent = "none"

        title = "none"

        line = f"{index}\t[{','.join(token)}]\t{comment}\t[{','.join(binary)}]\t[{','.join(multi)}]\t{parent}\t{title}\t\n"
        output.write(line)

output.close()
