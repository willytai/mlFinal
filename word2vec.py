from gensim.models import word2vec


def main():
	sentences = list(word2vec.LineSentence('dataset/segmentated/1_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/2_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/3_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/4_train_seg.txt'))
	sentences += list(word2vec.LineSentence('dataset/segmentated/5_train_seg.txt'))

	model = word2vec.Word2Vec(sentences, size=300)

	model.save('word2vec.model')

	####################
	## test similarity
	####################

	model = word2vec.Word2Vec.load('word2vec.model')
	
	test = "風火輪"

	buf = model.most_similar(test, topn=10)

	print (test)

	print ("相似詞前 10 排序")

	for item in buf:
		print ('{}, {}'.format(item[0], item[1]))


if __name__ == '__main__':
	main()