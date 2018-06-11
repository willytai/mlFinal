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
with open('dataset/testing_data.csv', encoding='cp950') as f:
    for line in f:
        if skip:
            skip = False
            continue

        line = filter_sentence(line).strip()
        line = re.sub(r' :', '\n', line)
        print(line)

        file.write(line)		
