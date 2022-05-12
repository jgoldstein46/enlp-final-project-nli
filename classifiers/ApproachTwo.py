from classifiers.Classifier import NLI_Classifier_Base
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

import transformers
from keras.models import Sequential
from keras.layers import Embedding,Dropout,Bidirectional, LSTM, Dense, TimeDistributed, Dropout,GlobalAveragePooling1D,GlobalMaxPooling1D,concatenate
from tensorflow.keras import Model
import tensorflow_hub as hub
import tensorflow_addons as tfa

transformers.logging.set_verbosity_error()


# TODO implement this class (or import it from huggingface if you like)
class BERT_NLI_Classifier(NLI_Classifier_Base):
    def __init__(self, params):
      super().__init__(params)
      self.hidden_size = params["hidden_size"]
      self.dropout = params["dropout"]
      self.vocab = params["vocab"]
      self.n_layers = params['n_layers']

      self.classifier = self.build_bert_baseline()

    def build_bert_baseline(self,max_length=128):
        
        # Encoded token ids
        input_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="input_ids"
        )
        # Attention masks 
        attention_masks = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="attention_masks"
        )
        # Token type ids 
        token_type_ids = tf.keras.layers.Input(
            shape=(max_length,), dtype=tf.int32, name="token_type_ids"
        )
        
        # Loading pretrained BERT model.
        bert_model = TFBertModel.from_pretrained("bert-base-uncased")

        bert_output = bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        
        sequence_output = bert_output.last_hidden_state
        pooled_output = bert_output.pooler_output
        
        # Add trainable layers 
        bi_lstm = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(self.hidden_size, return_sequences=True)
        )(sequence_output)
        
        # Pooling approach to bi_lstm sequence output.
        avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
        max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
        
        concat = tf.keras.layers.concatenate([avg_pool, max_pool])
        dropout = tf.keras.layers.Dropout(self.dropout)(concat)
        
        output = tf.keras.layers.Dense(3, activation="softmax")(dropout)
        model = tf.keras.models.Model(
            inputs=[input_ids, attention_masks, token_type_ids], outputs=output
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss="categorical_crossentropy",
            metrics=['accuracy', tf.keras.metrics.Recall(), tf.keras.metrics.Precision(), tfa.metrics.F1Score(num_classes=3,average='weighted')], 
            run_eagerly=False)


        model.summary()


        return model 