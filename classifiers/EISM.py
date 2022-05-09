from classifiers.Classifier import NLI_Classifier_Base
import numpy as np

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, Attention, concatenate, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.activations import softmax
import numpy as np
# from keras import backend as K
# from tensorflow.keras import Model
import tensorflow as tf
from math import exp
# from tensorflow import tfa


class EISM(NLI_Classifier_Base):
    def __init__(self, params):
        # pass
        super().__init__(params)
        # classifier = Sequential()
        # classifier.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout)))
        # our encoder
        self.encoder = Sequential()
        self.encoder.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout)))
        self.encoder.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout)))
        self.encoder.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout)))
        self.attention_layer = Attention(dropout=self.dropout)
        self.composition_layer = Sequential()
        self.composition_layer.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout)))
        self.composition_layer.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout)))
        self.composition_layer.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout)))
        # self.average_pooling = AveragePooling1D(pool_size=)
        self.MLP = Sequential()
        for _ in range(3): 
            self.MLP.add(Dense(self.hidden_size, activation='tanh'))
            self.MLP.add(Dropout(self.dropout))
        self.MLP.add(Dense(3, activation='softmax'))
        
        super().compile()
#     def train_step(self, data):
#         # Unpack the data. Its structure depends on your model and
#         # on what you pass to `fit()`.
#         x, y = data
#         print("calling model train_step with data: ", data)
#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)  # Forward pass
#             # Compute the loss value
#             # (the loss function is configured in `compile()`)
#             loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

#         # Compute gradients
#         trainable_vars = self.trainable_variables
#         gradients = tape.gradient(loss, trainable_vars)
#         # Update weights
#         self.optimizer.apply_gradients(zip(gradients, trainable_vars))
#         # Update metrics (includes the metric that tracks the loss)
#         self.compiled_metrics.update_state(y, y_pred)
#         # Return a dict mapping metric names to current value
#         return [m.result() for m in self.metrics]
    
    
        
    def call(self, inputs, **kwds):
        # inputs contains the premisis and hypothesis
        # print("calling eism with inputs", inputs)
        premise_emb = self.embedding_layer(inputs['premise'])
        hypothesis_emb = self.embedding_layer(inputs['hypothesis'])
        # premise_len = inputs['premise_len']
        # hypothesis_len = inputs['hypothesis_len']
        # premise_emb = self.embedding_layer(inputs[0])
        # hypothesis_emb = self.embedding_layer(inputs[1])
        #create hidden representations using encoder
        a_bar = self.encoder(premise_emb)
        b_bar = self.encoder(hypothesis_emb)
        # print(a_bar.shape, b_bar.shape)
        a_tilda = self.attention_layer([a_bar, b_bar])
        b_tilda = self.attention_layer([b_bar, a_bar])
        # print(a_tilda.shape, b_tilda.shape)
        m_a = concatenate([a_bar, a_tilda, a_bar - a_tilda, a_bar * a_tilda], axis=1)
        m_b = concatenate([b_bar, b_tilda, b_bar - b_tilda, b_bar * b_tilda], axis=1)
        # print(m_a.shape)
        v_a = self.composition_layer(m_a)
        v_b = self.composition_layer(m_b)
        # average_pooling = 
        v_a_ave, v_a_max = self.ave_and_max_pooling(v_a)
        v_b_ave, v_b_max = self.ave_and_max_pooling(v_b)
        # print(v_a_ave.shape)
        v = concatenate([v_a_ave, v_a_max, v_b_ave, v_b_max], axis=-1)
        # print(v.shape)
        output = self.MLP(v)
        # print(output.shape)
        return output # tf.squeeze(output, axis=[1]) 
        
        # print(v_a_ave.shape)
    def ave_and_max_pooling(self, v_x):
        # returns average and max pooling of v_x
        v_x_ave = GlobalAveragePooling1D()(v_x)
        v_x_max = GlobalMaxPooling1D()(v_x)
        return v_x_ave, v_x_max
        # return self.classifier(inputs)
#     def attention_layer(self, a_bar, b_bar): # a_bar and b_bar have size batch x seq len x hidden size --> 
#         # we want batch size x seq len of a_bar x 1 
#         # computes attention of a_bar on b_bar and returns it
#         # TODO: make it work with tensors 
#         a_tilda = tf.zeros(a_bar.shape[:2])
        
        
            
#         for i in range(a_tilda.shape[1]):
#             total = [tf.math.exp(tf.transpose(a_bar[:,i]) @ b_bar[:,k]) for k in range(b_bar.shape[1])]
#             print("total:", total)
            
#             a_tilda[:, i] = np.sum([exp(np.dot(a_bar[:,i].transpose(), b_bar[:,j]))/total * b_bar[:, j] for j in range(b_bar.shape[1])]) 
#         return a_tilda 
            
        