from classifiers.Classifier import NLI_Classifier_Base
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel


from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from tensorflow.keras import Model
import tensorflow_hub as hub

# TODO implement this class (or import it from huggingface if you like)
class BERT_NLI_Classifier(NLI_Classifier_Base):
    def __init__(self, params):
        hidden_size = params["hidden_size"]
        dropout = params["dropout"]
        self.vocab = params["vocab"]
        self.embedding_size = params["embedding_size"]
        self.n_layers = params['n_layers']

        """
        classifier = Sequential()
        embedding_layer = BERT_Wrapper(hidden_size)

        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
        input_segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_segment_ids")

        classifier.add(embedding_layer)
        #classifier.add(LSTM(hidden_size, return_sequences=False))
        
        # output layer
        classifier.add(Dense(3, activation='softmax'))
        # optimizer
        
        adam = tf.optimizers.Adadelta(clipvalue=0.5)
        classifier.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
        self.classifier = classifier
        """
        max_seq_length = 125

        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
        input_segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_segment_ids")

        bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2", trainable=True)

        pooled_output, _ = bert_layer([input_word_ids, input_mask, input_segment_ids])
        # pooled output is the embedding output for the '[CLS]' token that is dependant on all words of two sentences
        # and can be used for classfication purposes

        output_class = tf.keras.layers.Dense(units=3, activation='softmax')(pooled_output)
          
        classifier = tf.keras.Model(inputs=[input_word_ids, input_mask, input_segment_ids], outputs=output_class)

        optimizer = tf.keras.optimizers.Adam(lr=1e-5)
        classifier.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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