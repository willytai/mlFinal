import numpy as np
import sys, re

def filter_sentence(sentence):
    sentence = re.sub(r"\"", "", sentence)
    sentence = re.sub(r"\t", " ", sentence)
    sentence = re.sub(r"[0-9](?=:)", "", sentence)
    sentence = re.sub(r"^[0-9]{1,}", "", sentence)

    return sentence

file = open('dataset/training_data/test_full.txt', 'w+', encoding='utf-8')

skip = True
count = 0
with open('dataset/testing_data.csv', encoding='cp950') as f:
    for line in f:
        if skip:
            skip = False
            continue

        line = filter_sentence(line).strip()
        line = line.split(' :'); assert len(line) == 7, 'line {}, len = {}'.format(count+1, len(line))
        count += 1

        file.write('{}\n'.format('\n'.join(line)))

file.close()

assert count == 5060, 'missed some sentencs!! check errors in parse_testing.py!'

# there should be 35420 lines in test_full.txt