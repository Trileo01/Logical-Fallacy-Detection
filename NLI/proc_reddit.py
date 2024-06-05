import pandas as pd
import json


for split in ['train', 'dev', 'test']:
    with open(f'../BERT/data/processed/annotated_dataset/{split}.txt', 'r') as f:
        data = f.readlines()

    source = open(f"reddit/{split}.source", "w")
    target = open(f"reddit/{split}.target", "w")

    for d in data:
        col = d.split('\t')
        fallacy = list(set([f[1:-1] for f in col[4][1:-1].split(', ')]))
        print(fallacy)
        if len(fallacy) == 1:
            fallacy = fallacy[0]
        else:
            if fallacy[0] == 'none':
                fallacy = fallacy[1]
            else:
                fallacy = fallacy[0]
                
        comment = col[2]
        source.write(f"The text has logical fallacy </s> {comment}\n")
        target.write(f"{2 if fallacy != 'none' else 0}\n")
        source.write(f"The text does not have logical fallacy </s> {comment}\n")
        target.write(f"{2 if fallacy == 'none' else 0}\n")

    source.close()
    target.close()
