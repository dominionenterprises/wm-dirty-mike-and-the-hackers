from __future__ import print_function

import pandas as pd
import numpy as np
import os

import os
import numpy as np


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.models import load_model
import sys


class CentCom:

    def __init__(self, data_path= 'data/', file_name = 'first_100k.csv', label_col = 'stars',
                 col_names = ['stars', 'text', 'cool', 'useful', 'funny'], model_file_path):
        if os.path.isdir(data_path):
            self.data = pd.read_csv(data_path + file_name, header=None)
            self.data.fillna(value='', inplace=True)
            if col_names is not None:
                self.data.columns = col_names
        else:
            sys.exit(1)

        self.embedding_index = {}

        self.MAX_SEQUENCE_LENGTH = 1000
        self.MAX_NB_WORDS = 20000
        self.EMBEDDING_DIM = 100

        self.texts = []  # list of text samples
        self.labels_index = {}  # dictionary mapping label name to numeric id
        self.labels = []  # list of label ids
        self.text_data = None

        self.model = load_model(model_file_path) #loads a pre-trained keras model
        self.tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
        self.word_index = None

        if label_col in self.data.columns.values:
            self.label_col = label_col
        else:
            print("The label column you specified does not match any in the data set.")
            print("The columns are ", self.data.columns.values)

    def load_embedding(self, glove_dir='glove.6B/'):
        """ This script loads pre-trained word embeddings (GloVe embeddings).

        Sets the embedding_index attribute

        GloVe embedding data can be found at:
        http://nlp.stanford.edu/data/glove.6B.zip
        (source page: http://nlp.stanford.edu/projects/glove/)
        :type glove_dir: basestring"""

        f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
        for line in f:
            values = line.split()
            word = values[0]
            self.embeddings_index[word] = np.asarray(values[1:], dtype='float32')
        f.close()

    def _build_labels_dict(self):
        """ builds the labels_index dictionary by getting the number of classes
            from the data. """
        # infer number of classes, and their labels

        try:
            class_labels = list(self.data[self.label_col].unique())
        except KeyError:
            print "That column did not match any in the dataset, the columns are: ", self.data.columns.values
            return


        for i in xrange(len(class_labels)):
            self.labels_index[class_labels[i]] = i

        # self.labels_index['one'] = 0
        # self.labels_index['two'] = 1
        # self.labels_index['three'] = 2
        # self.labels_index['four'] = 3
        # self.labels_index['five'] = 4

    def _process_row(self):
        label_id = self.labels_index[row[0]]
        self.texts.append(row[1])
        self.labels.append(label_id)

    def preprocess(self):
        """ Populates the list of text samples and the label ids. Operates on the given data """

        self._build_labels_dict()
        self.data.apply(_process_row, axis=1)
        print('Found %s texts.' % len(self.texts))


    def tokenize(self, texts):
        """ tokenize the given texts, can be either a single text or a list of texts. """
        if not isinstance(texts, str) or not isinstance(texts, list):
            print("The sample you have provided is not correctly formatted. " \
                  "Must be a string or an array of strings.")
            return
        else:
            if isinstance(texts, str):
                texts = [texts]

        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        self.word_index = self.tokenizer.word_index
        print('Found %s unique tokens.' % len(self.word_index))
        return sequences

    def prep_data(self, ):
        """ pad the sequences to the maximum length, convert out labels array"""

        sequences = self.tokenize(texts=self.texts)
        self.text_data = pad_sequences(sequences, maxlen=self.MAX_SEQUENCE_LENGTH)

        self.labels = to_categorical(np.asarray(self.labels))
        print('Shape of data tensor:', self.text_data.shape)
        print('Shape of label tensor:', self.labels.shape)

        # split the data into a training set and a validation set
        indices = np.arange(data.shape[0])
        np.random.shuffle(indices)
        data = data[indices]
        labels = labels[indices]
        nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

        x_train = data[:-nb_validation_samples]
        y_train = labels[:-nb_validation_samples]
        x_val = data[-nb_validation_samples:]
        y_val = labels[-nb_validation_samples:]




