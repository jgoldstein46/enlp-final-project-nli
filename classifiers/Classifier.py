# base class for classifiers 
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras import Model
import tensorflow_addons as tfa
# from keras import backend as K
# from tensorflow.keras import Model



class NLI_Classifier_Base(Model):
    def __init__(self, params):
        super(NLI_Classifier_Base, self).__init__()
        # self.glove_file = params['glove-file']
        # self.vocab = params['vocab']
        # self.embedding_size = params['embedding-size']
        # self.embeddings_matrix = self.load_pretrained_embeddings()
        # self.model = None
        self.hidden_size = params["hidden_size"]
        # print(self.hidden_size)
        self.dropout = params["dropout"]
        self.vocab = params["vocab"]
        self.embedding_size = params["embedding_size"]
        self.glove_file = params["glove_file"]
        self.n_layers = params['n_layers']
        if params['classifier'] != "bert":
            embedding_matrix = self.load_pretrained_embeddings()
            self.embedding_layer = Embedding(len(self.vocab), self.embedding_size, 
                                        embeddings_initializer=tf.keras.initializers.Constant(embedding_matrix))
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0004)
    # override compile function to stop repeating duplicated code 
    def compile(self):
        super().compile(optimizer=self.optimizer, loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tfa.metrics.F1Score(num_classes=3,average='micro')], run_eagerly=False)

    def load_pretrained_embeddings(self):
        # return super().load_pretrained_embeddings()
        embedding_matrix = np.zeros((len(self.vocab), self.embedding_size))
        # filepath = os.path.join('..', self.glove_file)
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
        # print(np.where(~embedding_matrix.any(axis=1))[0].shape, embedding_matrix.shape)
        # print(embedding_matrix)
        return embedding_matrix
        
    
    
    