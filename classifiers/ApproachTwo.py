from classifiers.Classifier import NLI_Classifier_Base
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel


from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras import Model

# TODO implement this class (or import it from huggingface if you like)
class BERT_NLI_Classifier(NLI_Classifier_Base):
    def __init__(self, params):
        hidden_size = params["hidden_size"]
        dropout = params["dropout"]
        self.vocab = params["vocab"]
        self.embedding_size = params["embedding_size"]
        self.n_layers = params['n_layers']

        classifier = Sequential()
        embedding_layer = BERT_Wrapper(hidden_size)


        classifier.add(embedding_layer)
        classifier.add(LSTM(hidden_size, return_sequences=False))
        
        # output layer
        classifier.add(Dense(3, activation='softmax'))
        # optimizer
        
        adam = tf.optimizers.Adadelta(clipvalue=0.5)
        classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.classifier = classifier
        


class BERT_Wrapper(Model):

  def __init__(self,hidden_size):
    super(BERT_Wrapper, self).__init__()
    self.encoder = TFBertModel.from_pretrained("bert-base-uncased", trainable=False)
    self.dense = Dense(hidden_size)

  def call(self, inputs, **kwargs):
      outputs = self.encoder(inputs)
      last_hidden_states = outputs[0] # The last hidden-state is the first element of the output tuple
      output = self.dense(last_hidden_states)
      return output