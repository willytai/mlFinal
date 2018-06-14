import numpy as np
import sys

from util import cos_dist
from gensim.models.keyedvectors import KeyedVectors



test_file = 'dataset/segmentated/test_full_seg.txt'
# model_path = sys.argv[1]

test = [[], [], [], [], [], [], []]

count = 0
with open(test_file) as f:
	for line in f:
		print ('\rloading testing data {}/5060'.format(int((count+1)/7)), flush=True, end='')
		
		line = line.strip().split()
		test[count%7].append(line)
		count += 1

print ('')

# ensemble 3 models
models = []
for k in range(3):

	print ('loading word2vec from %s...' % sys.argv[k+1])
	w2v_model = KeyedVectors.load_word2vec_format(sys.argv[k+1], binary=True)
	vocab_size, Embedding_dim = w2v_model.wv.syn0.shape
	print ('vocab_size:   ', vocab_size)
	print ('Embedding_dim:', Embedding_dim)
	models.append(w2v_model)

result = []
for i in range(5060):
	print ('\rPredicting data {}/5060'.format(i+1), flush=True, end='')

	di = np.zeros(6)
	for w2v_model in models:
		ref = np.average(w2v_model[test[0][i]], axis=0)
		for idx in range(1, 7):
			choice = np.average(w2v_model[test[idx][i]], axis=0)
			di[idx-1] += cos_dist(ref, choice)
	
	di /= len(models)

	result.append(di.argmin())
			 
print ('')

print ('Writing to file...')
file = open('prediction.csv', 'w+')
file.write('{}\n'.format('id,ans'))
for i, choice in enumerate(result):
	file.write('{},{}\n'.format(i, choice))
file.close()



##########################################################
## this part tests the accuracy with respect to label.csv
##########################################################
ref = []
with open('label.csv') as f:
	for line in f:
		if line[0] == 'i':
			continue
		line = line.split(',')
		ref.append(int(line[-1]))

count = 0
all = 0
for i in range(5060):
	if ref[i] == 0:
		continue
	all += 1
	print ('\rEvaluating... %d' % all, flush=True, end='')
	if ref[i] == result[i]:
		count += 1
print ('\ntesting accuracy: %2f' % (count/all))

# record accuracy
# file = open('acc.csv', 'a+')
# file.write('{},{}\n'.format(model_path, count/all))
# file.close()