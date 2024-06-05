import pandas as pd

for split in ["train", "dev", "test"]:
    source = open(f"logic/{split}.source", "w")
    target = open(f"logic/{split}.target", "w")
    data = pd.read_csv(f"edu_{split}.csv")
    
    for index, row in data.iterrows():
        article = row['source_article'].replace('\n', ' ').replace('\r', ' ')
        source.write(f"The text has logical fallacy </s> {article}\n")
        target.write("2\n")
        source.write(f"The text does not have logical fallacy </s> {article}\n")
        target.write("0\n")

    source.close()
    target.close()
