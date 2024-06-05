import json
import sys
import shutil

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
    data = json.load(open(f'../{dataset}/test_{attack}.json', 'r'))
    shutil.copyfile(f'{dataset}/test.target', f'{dataset}_{attack}/test.target')

    output = open(f'{dataset}_{attack}/test.source', 'w')
    for i, d in enumerate(data):
        comment = d['comment'].replace('\n', ' ').replace('\r', ' ') + (d['attack'].replace('\n', ' ').replace('\r', ' ') if 'attack' in d else '')
        output.write(f'The text has logical fallacy </s> {comment}\n')
        output.write(f'The text does not have logical fallacy </s> {comment}\n')

    output.close()
