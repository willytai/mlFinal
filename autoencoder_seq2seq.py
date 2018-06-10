import sys
import numpy as np

from keras.models import Model, Sequential
from keras.layers import Dense, BatchNormalization, Dropout, LSTM, GRU, Embedding, Input
from keras.callbacks import EarlyStopping, ModelCheckpoint
from util import DataManager


def generate_vec_from_seq(train, mgr):
	model = Sequential()
	model.add(mgr.embedding_layer())
	for seq in train:
		vec = model.predict(seq)
		yield (vec, vec)

def CreateModel(mgr):
	###########################
	## paramater specification
	###########################
	max_len  = mgr.maxlen['train']
	droprate = 0.0
	loss     = 'mse'
	opt      = 'adam'

	seq_vec = Input(shape=(max_len, mgr.Embedding_dim))

	encoded = LSTM(1024, return_sequences=True, dropout=droprate)(seq_vec)
	encoded = LSTM(512,  return_sequences=True, dropout=droprate)(encoded)
	encoded = LSTM(128,  return_sequences=True, dropout=droprate)(encoded)
	encoded = LSTM(64,   return_sequences=True, dropout=droprate)(encoded)

	decoded = LSTM(128,  return_sequences=True, dropout=droprate)(encoded)
	decoded = LSTM(512,  return_sequences=True, dropout=droprate)(decoded)
	decoded = LSTM(1024, return_sequences=True, dropout=droprate)(decoded)

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

	model = CreateModel(mgr)

	generate_train = generate_vec_from_seq(train, mgr)
	generate_val   = generate_vec_from_seq(val, mgr)


	checkpoint = ModelCheckpoint('model/auto.h5', monitor='mse', verbose=1, save_best_only=True, mode='min')
	earlystop  = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

	batch_size = 256
	epochs = 10

	model.fit_generator(
		generator=generate_train,
		validation_data=generate_val,
		validation_steps=batch_size,
		samples_per_epoch=batch_size,
		epochs=epochs,
		use_multiprocessing=True,
		workers=-1,
		verbose=1,
		callbacks=[checkpoint, earlystop]
		)

if __name__ == '__main__':
	main()