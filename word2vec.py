from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors


def main():
	sentences  = list(word2vec.LineSentence('dataset/segmentated/1_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/2_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/3_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/4_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/5_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/test_seg.txt'))

	model = word2vec.Word2Vec(sentences, size=100, min_count=1, iter=50, workers=-1)

	print ('vocab_size:', model.wv.syn0.shape[0])

	word_vec = model.wv

	word_vec.save_word2vec_format('model/word2vec.bin', binary=True)

	####################
	## test similarity
	####################

	word_vectors = KeyedVectors.load_word2vec_format('model/word2vec.bin', binary=True)

	print ('successfully loaded')

	# model = word2vec.Word2Vec.load('word2vec.model')
	
	# test = "風火輪"

	# buf = model.most_similar(test, topn=10)

	# print (test)

	# print ("相似詞前 10 排序")

	# for item in buf:
	# 	print ('{}, {}'.format(item[0], item[1]))


if __name__ == '__main__':
	main()
