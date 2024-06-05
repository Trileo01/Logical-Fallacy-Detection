import pandas as pd

for split in ["train", "dev", "test"]:
    source = open(f"logicClimate/{split}.source", "w")
    target = open(f"logicClimate/{split}.target", "w")
    data = pd.read_csv(f"climate_{split}_mh.csv")
    
    for index, row in data.iterrows():
        if pd.isna(row['source_article']):
            continue
        article = row['source_article'].replace('\n', ' ').replace('\r', ' ')
        source.write(f"The text has logical fallacy </s> {article}\n")
        target.write("2\n")
        source.write(f"The text does not have logical fallacy </s> {article}\n")
        target.write("0\n")

    source.close()
    target.close()
