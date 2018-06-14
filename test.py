import numpy as np
import sys

from scipy import spatial
from gensim.models.keyedvectors import KeyedVectors


# load tfidf
tfidf = np.load('tfidf.npy').item()

def sentence_vector(text, wordmodel, tfidf=tfidf):
	vec = np.zeros(wordmodel.wv.syn0.shape[1])

	for word in text:
		vec += tfidf[word]*wordmodel[word]

	return vec

def cos_dist(ref, vec):
    ref = ref.reshape(-1)
    vec = vec.reshape(-1)
    
    return spatial.distance.cosine(ref,vec)    


test_file = 'dataset/segmentated/test_full_seg.txt'

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
for k in range(len(sys.argv)-1):

	print ('loading word2vec from %s...' % sys.argv[k+1])
	w2v_model = KeyedVectors.load_word2vec_format(sys.argv[k+1], binary=True)
	vocab_size, Embedding_dim = w2v_model.wv.syn0.shape
	print ('vocab_size:   ', vocab_size)
	print ('Embedding_dim:', Embedding_dim)
	models.append(w2v_model)

##############################################################
TFIDF = True # this is better
# TFIDF = False

result = []
for i in range(5060):
	print ('\rPredicting data {}/5060'.format(i+1), flush=True, end='')

	di = np.zeros(6)
	for w2v_model in models:
		if TFIDF == False:
			ref = np.average(w2v_model[test[0][i]], axis=0)
		else:
			ref = sentence_vector(test[0][i], w2v_model)
		for idx in range(1, 7):
			if TFIDF == False:
				choice = np.average(w2v_model[test[idx][i]], axis=0)
			else:
				choice = sentence_vector(test[idx][i], w2v_model)
			di[idx-1] += cos_dist(ref, choice)
	
	# di /= len(models)

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
# file.write('{},{}\n'.format(sys.argv[1], count/all))
# file.close()
