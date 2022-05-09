from classifiers.Classifier import NLI_Classifier_Base
import numpy as np
import keras
import tensorflow as tf
import numpy as np
from tensorflow import keras

from keras.layers import Embedding, Bidirectional, GRU, Dense, TimeDistributed, Dropout
from keras.layers import concatenate, Dense, Input, Dropout, TimeDistributed
from keras.layers.embeddings import Embedding
from tensorflow.keras.layers import BatchNormalization 
from keras.layers.wrappers import Bidirectional

import tensorflow_addons as tfa


class GRU_NLI_Classifier():
    def __init__(self, params):
        self.params = params
        self.hidden_size = params["hidden_size"]
        self.dropout = params["dropout"]
        self.vocab = params["vocab"]
        self.embedding_size = params["embedding_size"]
        self.glove_file = params["glove_file"]
        self.n_layers = params['n_layers']
        self.classifier = self.gru_new()  # higher accuracy
        # self.classifier = self.gru_model # lower accuracy

    def gru_new(self):
        LAYERS = 1
        MAX_LEN = 100

        timeDistr_layer = TimeDistributed(Dense(self.hidden_size, activation='relu'))
        # embedding_layer = Embedding(len(self.vocab), 100, input_length=MAX_LEN)
        embedding_matrix = self.load_pretrained_embeddings()
        embedding_layer = Embedding(len(self.vocab), self.embedding_size,
                                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix))

        # Layer 1 -- Two Input
        premise = Input(shape=(None,), dtype='int32')
        hypothesis = Input(shape=(None,), dtype='int32')
        # Layer 2 -- Two Embedding  
        prem = embedding_layer(premise)
        hypo = embedding_layer(hypothesis)
        # Layer 3 -- Two TimeDistributed
        prem = timeDistr_layer(prem)
        hypo = timeDistr_layer(hypo)
        # Layer 4-x -- Two biGRU
        if LAYERS > 1:
            for l in range(LAYERS - 1):
                bi_gru = Bidirectional(GRU(self.hidden_size, return_sequences=True))
                prem = BatchNormalization()(bi_gru(prem))
                hypo = BatchNormalization()(bi_gru(hypo))
        bi_gru = Bidirectional(GRU(self.hidden_size, return_sequences=False))
        prem = bi_gru(prem)
        hypo = bi_gru(hypo)
        prem = BatchNormalization()(prem)
        hypo = BatchNormalization()(hypo)
        # Layer 5 join together
        joint = concatenate([prem, hypo])
        # Layer 6-8 Dropout
        joint = Dropout(self.dropout)(joint)
        hidden_size = 2 * self.hidden_size
        for i in range(3):
            joint = Dense(hidden_size, activation='relu')(joint)
            joint = Dropout(self.dropout)(joint)
            joint = BatchNormalization()(joint)
            hidden_size=hidden_size/2
        # Layer 9 Output
        pred = Dense(3, activation='softmax')(joint)


        model = keras.models.Model([premise, hypothesis], pred)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(),
                               tfa.metrics.F1Score(num_classes=3)], run_eagerly=False)
        model.summary()
        return model

    def gru_model(self):
        embedding_matrix = self.load_pretrained_embeddings()
        embedding_layer = Embedding(len(self.vocab), self.embedding_size,
                                    embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix))

        model = keras.Sequential()
        model.add(embedding_layer)
        model.add(GRU(self.hidden_size, return_sequences=False))
        # output layer
        model.add(keras.layers.Dense(3, activation='softmax'))
        # optimizer
        adam = keras.optimizers.Adam(lr=1e-4)

        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def load_pretrained_embeddings(self):
        embedding_matrix = np.zeros((len(self.vocab), self.embedding_size))
        with open(self.glove_file, encoding='utf8') as f:
            for line in f:
                # Each line will be a word and a list of floats, separated by spaces.
                # If the word is in your vocabulary, create a numpy array from the list of floats.
                # Assign the array to the correct row of embedding_matrix.
                split_line = line.strip().split(" ")
                word = split_line[0]
                if word in self.vocab:
                    embeddings = np.array(split_line[1:])
                    embedding_matrix[self.vocab[word]] = embeddings

        embedding_matrix[self.vocab['[UNK]']] = np.random.randn(self.embedding_size)
        return embedding_matrix
