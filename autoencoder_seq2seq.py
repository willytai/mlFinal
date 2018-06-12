import sys
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LSTM, GRU, Embedding, Input, RepeatVector
from keras.callbacks import EarlyStopping, ModelCheckpoint
from util import DataManager

# train with 50% memory
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))


def to_vec(train, mgr, ratio=1):
	model = Sequential()
	model.add(mgr.embedding_layer())

	num = int(ratio*len(train))

	vec = model.predict(train[:num])

	return vec

def CreateModel(mgr):
	###########################
	## paramater specification
	###########################
	max_len  = mgr.maxlen['train']
	droprate = 0.0
	loss     = 'mse'
	opt      = 'adam'

	seq_vec = Input(shape=(max_len, mgr.Embedding_dim))
	encoded = GRU(mgr.Embedding_dim, return_sequences=True, dropout=droprate)(seq_vec)
	encoded = GRU(80,  return_sequences=True,  dropout=droprate)(encoded)
	encoded = GRU(64,  return_sequences=True,  dropout=droprate)(encoded)
	encoded = GRU(32,  return_sequences=False, dropout=droprate)(encoded)

	duplicate = RepeatVector(max_len)(encoded)

	decoded = GRU(64,  return_sequences=True, dropout=droprate)(duplicate)
	decoded = GRU(80,  return_sequences=True, dropout=droprate)(decoded)
	decoded = GRU(mgr.Embedding_dim, return_sequences=True, dropout=droprate)(decoded)

	encoder = Model(seq_vec, encoded)

	model   = Model(seq_vec, decoded)

	model.compile(loss=loss, optimizer=opt)
	model.summary()

	return model

def main():
	mgr = DataManager()

	mgr.add_data('train', 'dataset/segmentated')
	mgr.load_word2vec('model/word2vec.bin')
	mgr.to_sequence()

	train, val = mgr.split_data('train', 0.05)

	val   = to_vec(val, mgr, 1)
	train = to_vec(train, mgr, 1)

	model = CreateModel(mgr)

	##############
	## save model
	##############
	model_json = model.to_json()
	with open("model/auto.json", "w") as json_file:
		json_file.write(model_json)

	batch_size = 1024
	epochs = 20


	checkpoint = ModelCheckpoint('model/auto_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
	earlystop  = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

	model.fit(
		train, train,
		batch_size=batch_size,
		epochs=epochs,
		verbose=1,
		callbacks=[checkpoint, earlystop],
		validation_data=(val, val)
		)

if __name__ == '__main__':
	main()