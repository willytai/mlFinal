from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors


def main():
	sentences  = list(word2vec.LineSentence('dataset/segmentated/1_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/2_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/3_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/4_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/5_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/test_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/test_full_seg.txt'))

	sizes  = [50, 100, 150, 300]
	window = [15, 14, 13, 12, 11, 10, 9, 8]
	alphas = [0.01]
	its    = [80]

	for size in sizes:
		for win in window:
			for it in its:
				for alpha in alphas:
					print ('\rtraining word2vec: size {}, window {}'.format(size, win), end='', flush=True)
					model = word2vec.Word2Vec(sentences,
											sg=1,
											hs=1,
											alpha=alpha,
											min_alpha=0.00005,
											size=size,
											window=win,
											min_count=1,
											iter=it,
											workers=11)

					word_vec = model.wv

					word_vec.save_word2vec_format('model/word2vec{}_win{}_it{}_alp{}.bin'.format(size, win, it, alpha), binary=True)

	####################
	## test similarity
	####################

	# word_vectors = KeyedVectors.load_word2vec_format('model/word2vec.bin', binary=True)

	# print ('successfully loaded')


if __name__ == '__main__':
	main()
