from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
import sys

def main():
	sentences  = list(word2vec.LineSentence('dataset/segmentated/1_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/2_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/3_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/4_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/5_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/test_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/test_full_seg.txt'))

	# print (sentences[0]); sys.exit()

	sizes    = [50, 100, 150]
	window   = [15, 14, 13, 12, 11, 10, 9, 8, 7]
	negative = [5, 10, 15]
	sample   = [0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
	alphas   = [0.01]
	its      = [80]

	for size in sizes:
		for win in window:
			for neg in negative:
				for sam in sample:
					print ('training word2vec: size {}, window {}, negative {}'.format(size, win, neg))
					model = word2vec.Word2Vec(sentences,
											sg=1,
											hs=1,
											size=size,
											window=win,
											min_count=1,
											iter=80,
											negative=neg,
											sample=sam,
											workers=12)

					word_vec = model.wv

					word_vec.save_word2vec_format('new_model/word2vec{}_win{}_neg{}_sam{}.bin'.format(size, win, neg, sam), binary=True)

	####################
	## test similarity
	####################

	# word_vectors = KeyedVectors.load_word2vec_format('model/word2vec.bin', binary=True)

	# print ('successfully loaded')


if __name__ == '__main__':
	main()
