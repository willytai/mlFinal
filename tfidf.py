from gensim.models import TfidfModel, word2vec
from gensim import corpora, similarities
from gensim.models.keyedvectors import KeyedVectors

import numpy as np 

def main():
	sentences  = list(word2vec.LineSentence('dataset/segmentated/1_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/2_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/3_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/4_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/5_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/test_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/test_full_seg.txt'))

	dct = corpora.Dictionary(sentences)
	corpus = [dct.doc2bow(line) for line in sentences]  # convert dataset to BoW format

	tfidf = TfidfModel(corpus)

	corpus_tfidf = tfidf[corpus]
	d = {}
	for doc in corpus_tfidf:
		for id, value in doc:
			word = dct.get(id)
			d[word] = value

	np.save('tfidf.npy', d)


if __name__ == '__main__':
	main()
