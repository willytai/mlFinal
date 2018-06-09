import numpy as np
import sys, re

def filter_sentence(sentence):
	sentence = re.sub(r"\"", "", sentence)
	sentence = re.sub(r"[0-9]:", "", sentence)

	return sentence

file = open('dataset/training_data/test.txt', 'w+',encoding='utf-8')

skip = True
with open('dataset/testing_data.csv', encoding='cp950') as f:
	for line in f:
		if skip:
			skip = False
			continue

		line = line.split()[1:]
		line = [filter_sentence(l) for l in line]
		file.write('{}'.format("\n".join(line)))
		
