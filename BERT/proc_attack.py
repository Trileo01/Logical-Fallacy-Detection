import json
import sys
from nltk.tokenize import word_tokenize

attack = sys.argv[1]

escape_dict={'\a':r'\a',
             '\b':r'\b',
             '\c':r'\c',
             '\f':r'\f',
             '\n':r'\n',
             '\r':r'\r',
             '\t':r'\t',
             '\v':r'\v',
             '\'':r'\'',
             '\\':r'\\',
             '\"':r'\"'}


fallacy_map = {
    "appeal to authority": "authority",
    "appeal to majority": "population",
    "appeal to nature": "natural",
    "appeal to tradition": "tradition",
    "appeal to worse problems": "worse_problems",
    "false dilemma": "blackwhite",
    "hasty generalization": "hasty_generalization",
    "slippery slope": "slippery_slope"
}



def raw(text):
    """Returns a raw string representation of text"""
    new_string=''
    for char in text:
        try:
            new_string += escape_dict[char]
        except KeyError:
            new_string += char
    return new_string

for dataset in ['logic', 'logicClimate', 'reddit']:
    data = json.load(open(f'../../{dataset}/test_{attack}.json', 'r'))
    if dataset == 'reddit':
        with open('../processed/annotated_dataset/test.txt', 'r') as f:
            context = list(f.readlines())

    output = open(f'{dataset}_{attack}/test.txt', 'w')
    for i, d in enumerate(data):
        comment = d["comment"].replace('\n', ' ').replace('\r', ' ') + ' ' + (d['attack'].replace('\n', ' ').replace('\r', ' ') if 'attack' in d else '')
        token = word_tokenize(comment)
        token = [f"\'{raw(t)}\'" for t in token]
        if d["fallacy"] == "none":
            binary = "non_fallacy"
            fallacy = "none"
        else:
            if dataset == 'reddit':
                fallacy = fallacy_map[d['fallacy']]
            else:
                fallacy = "none"
            binary = "fallacy"
        

        binary = [f"\'{binary}\'"] * len(token)
        multi = [f"\'{fallacy}\'"] * len(token)

        if dataset == 'reddit':
            parent = context[i].split('\t')[5]
            title = context[i].split('\t')[6]
        else:
            parent = 'none'
            title = 'none'

        
        line = f"{i}\t[{','.join(token)}]\t{comment}\t[{','.join(binary)}]\t[{','.join(multi)}]\t{parent}\t{title}\t\n"
        output.write(line)

    output.close()
