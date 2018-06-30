from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from random import shuffle
from keras.models import Model
from keras.layers import LSTM, Dense, Input

import numpy as np
import sys
import util as utl

# specify device
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# train with 50% memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.Session(config=config))



MAX_LENGTH = 22



word_model = KeyedVectors.load_word2vec_format('model/word2vec100.bin', binary=True)

vocab_size, embedding_dim = word_model.wv.syn0.shape

print (vocab_size, ',', embedding_dim)

Id2Word = {word_model.wv.vocab[fuck].index: fuck for fuck in word_model.wv.vocab}


def getVec(word):
	try:
		word_vector = word_model[word]
	except KeyError:
		raise KeyError
	return word_vector

def preprocess():

	encoder_line_in  = list(word2vec.LineSentence('dataset/conversation/encoder.input'))
	decoder_line_in  = list(word2vec.LineSentence('dataset/conversation/decoder.input'))
	decoder_line_out = list(word2vec.LineSentence('dataset/conversation/decoder.output'))

	c = list(zip(encoder_line_in, decoder_line_in, decoder_line_out))
	shuffle(c)
	encoder_line_in, decoder_line_in, decoder_line_out = zip(*c)

	return encoder_line_in, decoder_line_in, decoder_line_out

def reply(text, encoder, decoder):
	seq = np.zeros((1, MAX_LENGTH, embedding_dim))
	for t in range(len(text)):
		time = MAX_LENGTH-1-t
		word = text[t]

		if time < 0:
			break
		try:
			word_vector = getVec(word)
		except KeyError:
			print ('{} is out of vocabulary'.format(word))
			continue

		seq[0][time] = word_vector


	# # test
	# tmp = np.zeros(20)
	# for i in range(20):
	# 	tmp[i] = np.argmax(seq[0, i, :])

	# print ('idx: ', tmp)
	# tmp = [dict_inv[t] for t in tmp]
	# print ('text: ', tmp); sys.exit()

	encoded = encoder.predict(seq)
	initial_state = encoded

	target_seq = np.zeros((1, 1, embedding_dim))
	target_seq[0][0] = getVec('卍')

	GO = True
	reply = ''
	while GO:
		vector_out, h, c = decoder.predict([target_seq]+initial_state)

		word_id = np.argmax(vector_out[0, -1, :])
		word    = Id2Word[word_id]
		reply  += word

		if word == '乂' or len(reply) >= MAX_LENGTH:
			GO = False

		target_seq = np.zeros((1, 1, embedding_dim))
		target_seq[0][0] = getVec(word)

		initial_state = [h, c]

	return reply

def generator(encoder_line_in, decoder_line_in, decoder_line_out, batch_size):
	batch_num = int(len(encoder_line_in) / batch_size)
	curr_batch = 0

	while True:
		
		if curr_batch == batch_num:
			c = list(zip(encoder_line_in, decoder_line_in, decoder_line_out))
			shuffle(c)
			encoder_line_in, decoder_line_in, decoder_line_out = zip(*c)
			curr_batch = 0

		line_id     = curr_batch*batch_size
		encoder_in  = np.zeros((batch_size, MAX_LENGTH, embedding_dim))
		decoder_in  = np.zeros((batch_size, MAX_LENGTH, embedding_dim))
		decoder_out = np.zeros((batch_size, MAX_LENGTH, vocab_size))

		for i in range(batch_size):

			# process encoding input
			# ** pad zero from the front and reverse the seq **
			for t in range(len(encoder_line_in[line_id])):
				time = MAX_LENGTH-1-t
				word = encoder_line_in[line_id][t]
				if time < 0:
					break
				try:
					word_vector = getVec(word)
				except KeyError:
					print ('\r{} is OOV'.format(word), end='', flush=True)
					continue

				encoder_in[i][time] = word_vector

			# process decoding input
			for t, word in enumerate(decoder_line_in[line_id]):
				if t > MAX_LENGTH-1:
					break
				try:
					word_vector = getVec(word)
				except KeyError:
					print ('\r{} is OOV'.format(word), end='', flush=True)
					continue

				decoder_in[i][t] = word_vector

			# process decoding output
			for t, word in enumerate(decoder_line_out[line_id]):
				if t > MAX_LENGTH-1:
					break

				if word not in word_model.wv.vocab:
					continue

				word_id = word_model.wv.vocab[word].index
				decoder_out[i][t][word_id] = 1

			# process next line
			line_id += 1

		# move the window
		curr_batch += 1



		### check ###
		# for i in range(len(encoder_in)):
		# 	print ('en', encoder_in[i, -5:, :])
		# 	print ('de', decoder_in[i, :5, :])
		# 	tmp = []
		# 	for j in range(5):
		# 		tmp.append(np.argmax(decoder_out[i, j, :]))
		# 	print ('de', tmp); sys.exit()

		# 	tmp1 = np.zeros(MAX_LENGTH, embedding_dim)
		# 	tmp2 = np.zeros(MAX_LENGTH, embedding_dim)
		# 	tmp3 = np.zeros(MAX_LENGTH, embedding_dim)
		# 	for j in range(MAX_LENGTH):
		# 		tmp1[j] = encoder_in[i, j, :]
		# 		tmp2[j] = decoder_in[i, j, :]
		# 		tmp3[j] = decoder_out[i, j, :]

		# 	t1 = [Vec2Word[idx] for idx in tmp1]
		# 	t2 = [Vec2Word[idx] for idx in tmp2]
		# 	t3 = [Vec2Word[idx] for idx in tmp3]

		# 	print ('==== after ====')
		# 	print (" ".join(t1),'\n', " ".join(t2),'\n', " ".join(t3),'\n',); sys.exit()
			

		# print (encoder_in.shape)
		# print (decoder_in.shape)
		# print (decoder_out.shape)
		# sys.exit()
		yield ([encoder_in, decoder_in], decoder_out)

def CreateModel():
	##############
	## paramaters
	##############
	hidden_num_rnn = 256
	hidden_num_dnn = vocab_size



	# Define an input sequence and process it.
	encoder_inputs = Input(shape=(None, embedding_dim))
	encoder = LSTM(hidden_num_rnn, return_state=True)
	_, state_h, state_c = encoder(encoder_inputs)

	# only keep the states.
	encoder_states = [state_h, state_c]

	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(None, embedding_dim))

	decoder_lstm = LSTM(hidden_num_rnn, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
	decoder_dense = Dense(hidden_num_dnn, activation='softmax')
	decoder_outputs = decoder_dense(decoder_outputs)

	# training model
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

	# encoder
	encoder = Model(encoder_inputs, encoder_states)

	# decoder
	decoder_h = Input(shape=(hidden_num_rnn,))
	decoder_c = Input(shape=(hidden_num_rnn,))
	decoder_states_input = [decoder_h, decoder_c]
	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_input)
	decoder_states = [state_h, state_c]
	decoder_outputs = decoder_dense(decoder_outputs)
	decoder = Model([decoder_inputs] + decoder_states_input, [decoder_outputs] + decoder_states)	

	return model, encoder, decoder	

def main():

	##############
	## paramaters
	##############
	batch_size = 256
	epochs     = 5

	encoder_line_in, decoder_line_in, decoder_line_out = preprocess()

	train_generator = generator(encoder_line_in, decoder_line_in, decoder_line_out, batch_size)

	model, encoder, decoder = CreateModel()

	###############
	## save model
	###############
	model_json = model.to_json()
	with open("model/model.json", "w") as json_file:
		json_file.write(model_json)

	model_json = encoder.to_json()
	with open("model/encoder.json", "w") as json_file:
		json_file.write(model_json)

	model_json = decoder.to_json()
	with open("model/decoder.json", "w") as json_file:
		json_file.write(model_json)


	model.summary()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

	model.fit_generator(train_generator,
						verbose=1,
						steps_per_epoch=int(len(encoder_line_in)/batch_size),
						epochs=epochs,
						workers=8)


	model.save_weights('model/final_weight.hdf5')
	encoder.save_weights('model/final_encoder_weight.hdf5')
	decoder.save_weights('model/final_decoder_weight.hdf5')


	############### test ###############

	test1 = utl.segmentation('你太抬舉我了我哪有能力籌資呢我還太年輕根本沒有人脈')
	test2 = utl.segmentation('她媽沒再打電話來應該沒事了你啊') 
	test3 = utl.segmentation('你現在是台灣')
	test  = [test1, test2, test3]


	print ('=========test=========')
	result = []
	for t in test:
		result.append(reply(t, encoder, decoder))
	for i, r in enumerate(result):
		print (test[i], ' -> ', r)

if __name__ == '__main__':
	main()