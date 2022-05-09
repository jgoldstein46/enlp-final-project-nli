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
      super().__init__(params)
      hidden_size = params["hidden_size"]
      dropout = params["dropout"]
      self.vocab = params["vocab"]
      self.embedding_size = params["embedding_size"]
      self.n_layers = params['n_layers']

      self.classifier = self.build_bert_baseline()




        

    def build_bert_baseline(self,max_len=128):
          bert_layer = TFBertModel.from_pretrained('bert-base-uncased')
          input_word_ids = tf.keras.Input(shape = (max_len,),dtype =tf.int32, name='input_word_ids')
          input_masks = tf.keras.Input(shape = (max_len,),dtype =tf.int32, name='input_masks')
          input_type_ids = tf.keras.Input(shape = (max_len,),dtype =tf.int32, name='input_type_ids')
          
          sequence_output = bert_layer([input_word_ids, input_masks, input_type_ids])[0]
          #sequence_output = clf_output[0]
          output = tf.keras.layers.Dense(3, activation='softmax')(sequence_output[:,0,:])
          
          model = tf.keras.Model(inputs=[input_word_ids,input_masks,input_type_ids],outputs=output)
          model.compile(tf.keras.optimizers.Adam(lr=3e-5),
                        loss='categorical_crossentropy',metrics=['accuracy'])
          
          print(model.summary())
          
          return model