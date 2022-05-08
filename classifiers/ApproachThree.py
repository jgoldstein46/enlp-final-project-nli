from classifiers.Classifier import NLI_Classifier_Base
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, concatenate

import numpy as np
# from keras import backend as K
# from tensorflow.keras import Model
import tensorflow as tf
import os

class LSTM_NLI_Classifier(NLI_Classifier_Base):
    def __init__(self, params):
        # hidden_size = params["hidden_size"]
        # dropout = params["dropout"]
        # self.vocab = params["vocab"]
        # self.embedding_size = params["embedding_size"]
        # self.glove_file = params["glove_file"]
        # self.n_layers = params['n_layers']
        # embedding_matrix = self.load_pretrained_embeddings()
        # embedding_layer = Embedding(len(self.vocab), self.embedding_size, 
        #                             embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix))
        super().__init__(params)
        classifier = Sequential()
        classifier.add(self.embedding_layer)
        classifier.add(Dropout(self.dropout))
        for _ in range(self.n_layers - 1):
            classifier.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout)))
            classifier.add(TimeDistributed(Dense(self.hidden_size, activation='softmax')))
        classifier.add(Bidirectional(LSTM(self.hidden_size, return_sequences=False, dropout=self.dropout)))
        classifier.add(Dense(3, activation='softmax'))
        self.classifier = classifier
        
        # self.compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        self.compile()
    
#     def load_pretrained_embeddings(self):
#         # return super().load_pretrained_embeddings()
#         embedding_matrix = np.zeros((len(self.vocab), self.embedding_size))
#         # filepath = os.path.join('..', self.glove_file)
#         with open(self.glove_file, encoding='utf8') as f:
#             for line in f:
#                 # Each line will be a word and a list of floats, separated by spaces.
#                 # If the word is in your vocabulary, create a numpy array from the list of floats.
#                 # Assign the array to the correct row of embedding_matrix.
#                 split_line = line.strip().split(" ")
#                 word = split_line[0]
#                 if word in self.vocab:
#                     embeddings = np.array(split_line[1:])
#                     embedding_matrix[self.vocab[word]] = embeddings

#         embedding_matrix[self.vocab['[UNK]']] = np.random.randn(self.embedding_size)
#         # print(np.where(~embedding_matrix.any(axis=1))[0].shape, embedding_matrix.shape)
#         # print(embedding_matrix)
#         return embedding_matrix

    def call(self, inputs, **kwds):
        seq = concatenate((inputs['premise'], inputs['hypothesis']), axis=-1)
        # print(seq)
        return self.classifier(seq)