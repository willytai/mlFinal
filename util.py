import re
import os, sys
import numpy as np
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle as pkl

class DataManager():
    def __init__(self):
        self.vec = {}
        self.data = {}
        self.maxlen = {} 

    def add_data(self,name, data_path):
        self.maxlen[name] = 0
        if name=='train':
            X = []
            file_path = ['1_train_seg.txt', '2_train_seg.txt', '3_train_seg.txt', '4_train_seg.txt', '5_train_seg.txt', 'test_seg.txt']
            file_path = [os.path.join(data_path, path) for path in file_path]
            for path in file_path:
                print ('reading data from %s...'%path)
                with open(path,'r') as f:
                    for line in f:
                        line = line.strip().split()
                        if len(line) > self.maxlen[name]:
                            self.maxlen[name] = len(line)
                        X.append(line)
            self.data[name] = [X]

        else:
            print ('reading data from %s...'%data_path)
            X = [[], [], [], [], [], [], []]
            i = 0
            with open(data_path,'r') as f:
                for line in f:
                    line = line.strip().split()
                    X[i%7].append(line)
                    i += 1
                    if len(line) > self.maxlen[name]:
                        self.maxlen[name] = len(line)
            self.data[name] = X 
        print ('maxlen: %d'%self.maxlen[name])

    def to_sequence(self):
        for key in self.data:
            for idx in range(len(self.data[key])):
                sentences = self.data[key][idx]
                tmp = []
                for i in range(len(sentences)):
                    print ('\rConverting {} to sequences... {}'.format(key, i+1), flush=True, end='')
                    st = []
                    for j in range(len(sentences[i])):
                        if sentences[i][j] in self.w2v_model.wv.vocab:
                            #print(self.w2v_model.wv.vocab[self.data[key][0][i][0]].index)
                            st.append(self.w2v_model.wv.vocab[sentences[i][j]].index)
                    tmp.append(st)
                self.data[key][idx] = np.array(pad_sequences(tmp, maxlen=self.maxlen[key]))
                print ('')
    
    def get_data(self,name='', all=False):
        if all:
            data = []
            for key in self.data:
                data += self.data[key][0]
            return data 
        elif name == 'test':
            return np.array(self.data['test'])
        return np.array(self.data[name][0])

    def load_word2vec(self, data_path):
        print ('load word2vec from %s...'%data_path)
        self.w2v_model = KeyedVectors.load_word2vec_format(data_path, binary=True)
        self.vocab_size, self.Embedding_dim = self.w2v_model.wv.syn0.shape
        print ('vocab_size:   ', self.vocab_size)
        print ('Embedding_dim:', self.Embedding_dim)

    def embedding_layer(self):
        return self.w2v_model.wv.get_keras_embedding(False)

    def shuffle_data(self, name):
        p = np.random.permutation(len(self.data[name][0]))
        self.data[name][0] = self.data[name][0][p] 

    def split_data(self, name, ratio):
        X = self.data[name][0]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return X[val_size:], X[:val_size]

if __name__=='__main__':
    dm = DataManager()
    #dm.add_data('train', 'dataset/segmentated')
    dm.add_data('test', 'dataset/segmentated/test_seg.txt')
    dm.load_word2vec('model/word2vec.bin')
    dm.to_sequence()
    #dm.shuffle_data('train')
    #X_train, X_val = dm.split_data('train', 0.1)
    #print(dm.data['test'][0])
    #print(dm.data['test'][1])

