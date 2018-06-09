from seq2vec import Seq2VecR2RWord
from gensim.models import word2vec
from seq2vec.word2vec import GensimWord2vec

import sys

def read():
	data   = []
	files  = []
	files.append(sys.argv[2])
	files.append(sys.argv[3])
	files.append(sys.argv[4])
	files.append(sys.argv[5])
	files.append(sys.argv[6])
	files.append(sys.argv[7])

	for file in files:
		print ('Reading from {}'.format(file), end='')
		count = 0
		with open(file, encoding='utf-8') as f:
			for line in f:
				data.append(line.split())
				count += 1
		print ('	{} sentences loaded'.format(count))
	return data



word_model_path = sys.argv[1]

# word_model = word2vec.Word2Vec.load(word_model_path)
word_model = GensimWord2vec(word_model_path)

train_seq = read()
print ('Number of training data:', len(train_seq))
test_seq = ['我們', '拾獲', '妳', '妹妹', '物品', '的', '地方' ]

transformer = Seq2VecR2RWord(
      word2vec_model=word_model,
      max_length=20,
      latent_size=100,
      encoding_size=50,
      learning_rate=0.05
)

transformer.fit(train_seq)
result = transformer.transform(test_seq)
print ('test: {}'.format(" ".join(test_seq)))
print ('next: {}'.format(" ".join(result)))