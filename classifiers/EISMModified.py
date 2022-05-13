from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout, concatenate
from classifiers.Classifier import NLI_Classifier_Base
import numpy as np
from keras.models import Sequential
from keras import layers
from tensorflow.keras.layers import BatchNormalization

class ModifiedEISM(NLI_Classifier_Base):
    def __init__(self, params):
        super().__init__(params)
        self.encoder = Sequential()
        self.encoder.add(Bidirectional(LSTM(self.hidden_size, return_sequences=True, dropout=self.dropout)))
        self.encoder.add(BatchNormalization())
        self.self_att = TransformerBlock()
        self.composition = Bidirectional(LSTM(self.hidden_size, return_sequences=False, dropout=self.dropout))
        self.MLP = Sequential()

        for _ in range(3): 
            self.MLP.add(Dense(self.hidden_size, activation='tanh'))
            self.MLP.add(Dropout(self.dropout))

        self.MLP.add(Dense(3, activation='softmax'))
        
        super().compile()

    def call(self, inputs, **kwds):
        prem_emb = self.embedding_layer(inputs['premise'])
        hyp_emb = self.embedding_layer(inputs['hypothesis'])

        prem_enc = self.encoder(prem_emb)
        hyp_enc = self.encoder(hyp_emb)

        combined = concatenate([prem_enc, hyp_enc], axis=1)
        print(combined.shape)
        att_out = self.self_att(combined)
        print(att_out.shape)
        # composition layer 
        cls = self.composition(att_out)

        # classification
        output = self.MLP(cls)

        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)