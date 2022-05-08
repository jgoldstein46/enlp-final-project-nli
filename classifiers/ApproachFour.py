from classifiers.Classifier import NLI_Classifier_Base
import numpy as np
import tensorflow as tf
import numpy as np
from tensorflow import keras
import os
from keras.layers import Embedding, Bidirectional, GRU, Dense, TimeDistributed, Dropout
import matplotlib.pyplot as plt

class GRU_NLI_Classifier(NLI_Classifier_Base):
    def __init__(self, params):
        self.hidden_size = params["hidden_size"]
        dropout = params["dropout"]
        self.vocab = params["vocab"]
        self.embedding_size = params["embedding_size"]
        self.glove_file = params["glove_file"]
        self.n_layers = params['n_layers']

        self.classifier = self.gru_model()



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

