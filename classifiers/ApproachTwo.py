from classifiers.Classifier import NLI_Classifier_Base
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel


from keras.models import Sequential
from keras.layers import Embedding,Dropout,Bidirectional, LSTM, Dense, TimeDistributed, Dropout,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
from tensorflow.keras import Model
import tensorflow_hub as hub

# TODO implement this class (or import it from huggingface if you like)
class BERT_NLI_Classifier(NLI_Classifier_Base):
    def __init__(self, params):
      super().__init__(params)
      hidden_size = params["hidden_size"]
      self.dropout = params["dropout"]
      self.vocab = params["vocab"]
      self.embedding_size = params["embedding_size"]
      self.n_layers = params['n_layers']
      self.hidden_size = params["hidden_size"]

      self.classifier = self.build_bert_baseline()

    def build_bert_baseline(self,max_len=128):
          bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
          
          input_word_ids = tf.keras.Input(shape = (max_len,),dtype =tf.int32, name='input_word_ids')
          input_masks = tf.keras.Input(shape = (max_len,),dtype =tf.int32, name='input_masks')
          input_type_ids = tf.keras.Input(shape = (max_len,),dtype =tf.int32, name='input_type_ids')
          
          bert_output = bert_layer(
              input_word_ids, attention_mask=input_masks, token_type_ids=input_type_ids)
          
          sequence_output = bert_output.last_hidden_state
          pooled_output = bert_output.pooler_output
          
          # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
          bi_lstm = Bidirectional(LSTM(self.hidden_size, return_sequences=True))(sequence_output)

          avg_pool = GlobalAveragePooling1D()(bi_lstm)
          max_pool = GlobalMaxPooling1D()(bi_lstm)
          concat = concatenate([avg_pool, max_pool])
          
          dropout = Dropout(self.dropout)(concat)
          #sequence_output = clf_output[0]
          output = Dense(3, activation='softmax')(dropout)
          
          
          model = Model(inputs=[input_word_ids,input_masks,input_type_ids],outputs=output)
          model.compile(
              optimizer=tf.keras.optimizers.Adam(),
              loss="categorical_crossentropy",
              metrics=["acc"])

          
          print(model.summary())
          
          return model