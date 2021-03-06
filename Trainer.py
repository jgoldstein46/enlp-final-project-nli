from DataFrame import DataFrame
# from classifiers.Classifier import NLI_Classifier
from classifiers.ApproachOne import NLI_Baseline
from classifiers.ApproachTwo import BERT_NLI_Classifier
from classifiers.ApproachFour import GRU_NLI_Classifier
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import math
import re
from tensorflow import keras
from datetime import datetime
from transformers import GPT2Tokenizer
import tensorflow_addons as tfa


SEP = '[SEP]'
CLS = '[CLS]'
START = '<s>'
END = '</s>'
UNK = '[UNK]'
PAD = '[PAD]'


class NLI_Trainer:
    def __init__(self, params):
        print("Creating the trainer")
        self.params = params

        # train file will be used for dev and testing after it is split
        self.train_file = params['train']
        df = DataFrame(pd.read_csv(self.train_file))
        # start out with only english
        english_df = df.english_df
        self.df = english_df
        ########################
        # Train test split     #
        ########################
        # print(english_df.head())
        # first split up between train and dev/test
        self.train_data, dev_data, self.train_labels, dev_labels = train_test_split(
            english_df[['id', 'premise', 'hypothesis', 'lang_abv', 'language']],
            english_df[['label']], test_size=0.3, random_state=self.params['seed'])
        
        # then reserve two thirds of the dev/test data for testing (20% of the total data)
        # print(dev_data.shape, dev_labels.shape)
        self.dev_data, self.test_data, self.dev_labels, self.test_labels = train_test_split(dev_data,
                                                                                            dev_labels, test_size=0.67,
                                                                                            random_state=self.params[
                                                                                                'seed'])

        classifier = self.params['nli_classifier_class']
        self.baseline = classifier == NLI_Baseline
        self.use_bert = classifier == BERT_NLI_Classifier
        self.use_gru = classifier == GRU_NLI_Classifier
        
        if self.params['classifier'] == 'gpt2': 
            batch_generator = self.GPT2BatchGenerator
            self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2", num_labels=3)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.params['classifier_params']['tokenizer'] = self.tokenizer
            
        
        elif self.params['classifier'] == "bert":
            batch_generator = self.BERTBatchGenerator
        
        else:
            batch_generator = self.BatchGenerator

        # create the vocab using tokenizer if necessary
        self.vocab = self.create_vocab(self.train_data)
        self.params['classifier_params']['vocab'] = self.vocab

        self.nli_classifier = classifier(self.params['classifier_params'])


        self.train_batch_generator = batch_generator(self, self.train_data, self.train_labels,
                                                     self.params['batch_size'])
        self.dev_batch_generator = batch_generator(self, self.dev_data, self.dev_labels, 1)
        
        self.test_batch_generator = batch_generator(self, self.test_data, self.test_labels, 1)

            
    class GPT2BatchGenerator(Sequence):
        def __init__(self, trainer, data_df, labels_df, batch_size): 
            self.data_l = [[prem, hyp] for prem, hyp in zip(data_df['premise'].tolist(), data_df['hypothesis'].tolist())]
            
            # self.label_a = labels_df['label'].tolist()
            self.label_a = [trainer.one_hot_encode_label(label) for label in labels_df['label'].tolist()]
            # [trainer.one_hot_encode_label(label) for label in labels_df['label'].tolist()]
            
            self.batch_size = batch_size
            self.trainer = trainer
            self.tokenizer = trainer.tokenizer
            # self.tokenizer.pad_token = self.tokenizer.eos_token
            # print(self.data_l, self.label_a, self.tokenizer)
            
        def __len__(self):
            return math.ceil(len(self.label_a) / self.batch_size)
        
        def __getitem__(self, idx):
            # return 
            # print(self.data_l[idx*self.batch_size:(idx+1)*self.batch_size])
            input_batch = self.tokenizer(text=self.data_l[idx*self.batch_size:(idx+1)*self.batch_size], padding=True, return_token_type_ids=True, return_attention_mask=True, verbose=False, return_tensors='tf', truncation=True, is_split_into_words=False)
            print(type(input_batch))
            # self.data_l[idx*self.batch_size:(idx+1)*self.batch_size], text_pair=self.data_l[idx*self.batch_size:(idx+1)*self.batch_size]
            # [["this is some text", 'this is some other text'], 
                                         # ["this is some different text", 'this is some more special text']]
            # print(input_batch['input_ids'])
#             return_token_type_ids=True, return_attention_mask=True, verbose=False
            # input_batch['labels'] = np.array(self.label_a[idx*self.batch_size:(idx+1)*self.batch_size])
            batch_y = np.array(self.label_a[idx * self.batch_size:(idx + 1) * self.batch_size])
            print('batch_y:', batch_y)
            return (dict(input_batch), batch_y)
        
    class BatchGenerator(Sequence):
        def __init__(self, trainer, data_df, labels_df, batch_size):
            self.hypothesis_a = [trainer.vectorize_sequence(seq) for seq in
                                 trainer.preprocess_sentences(data_df['hypothesis'].tolist())]
            # print("after initializing hypothesis_a, here is its length:", len(self.hypothesis_a))
            self.premise_a = [trainer.vectorize_sequence(seq) for seq in
                              trainer.preprocess_sentences(data_df['premise'].tolist())]

            self.label_a = [trainer.one_hot_encode_label(label) for label in labels_df['label'].tolist()]
            self.batch_size = batch_size
            self.trainer = trainer

        def __len__(self):
            return math.ceil(len(self.label_a) / self.batch_size)

        def __getitem__(self, idx):
            premise = self.get_x_batch(self.premise_a, idx)
            hypothesis = self.get_x_batch(self.hypothesis_a, idx)
            # print()
            batch_y = np.array(self.label_a[idx * self.batch_size:(idx + 1) * self.batch_size])
            inputs = {
                'premise': premise,
                'hypothesis': hypothesis
            }
            # print("calling get item with inputs, y: ", inputs, batch_y, sep='\n')
            return (inputs, batch_y)

        def get_x_batch(self, a, idx):
            return np.array(self.trainer.pad_sequences(a[idx * self.batch_size:(idx + 1) * self.batch_size],
                                                       self.trainer.vocab[PAD]))

    def batch_generator_nli(self, data_df, labels_df, batch_size=1):
        hypothesis_a = self.preprocess_sentences(data_df['hypothesis'].tolist())
        premise_a = self.preprocess_sentences(data_df['premise'].tolist())
        label_a = labels_df['label'].tolist()

        while True:
            batch_x = [[], []]
            batch_y = []
            for i, (premise, hypothesis) in enumerate(zip(premise_a, hypothesis_a)):
                label = label_a[i]
                # if self.use_bert:
                #     # add indexes of cls and sep tokens to make proper sentence form for BERT
                #     seq = [101] + self.vectorize_sequence(premise) + [102] + self.vectorize_sequence(hypothesis) + [102]

                # other methods will have sentenfes separated by START and END tokens
                # seq = self.vectorize_sequence(premise) + self.vectorize_sequence(hypothesis)
                # print(seq, self.unvectorize_sequence(seq))
                # print(premise, hypothesis, self.one_hot_encode_label(label))
                premise_ids, hypothesis_ids = self.vectorize_sequence(premise), self.vectorize_sequence(hypothesis)
                # batch_x.append(seq)
                batch_x[0].append(premise_ids)
                batch_x[1].append(hypothesis_ids)
                # premise has to be of size
                # batch_size x premise len padded
                # batch_x.append()
                batch_y.append(self.one_hot_encode_label(label))

                if len(batch_x[0]) >= batch_size:
                    # print("after going through batch, here it is: ")
                    # batch_x = self.pad_sequences(batch_x, self.vocab[PAD])
                    premises = np.array(self.pad_sequences(batch_x[0], self.vocab[PAD]))
                    hypotheses = np.array(self.pad_sequences(batch_x[1], self.vocab[PAD]))
                    inputs = {
                        'premise': premises,
                        'hypothesis': hypotheses
                    }
                    # inputs = [premises, hypotheses]

                    # batch_x, batch_y = np.array(batch_x, np.array(batch_y)
                    # print(premises.shape, hypotheses.shape)
                    batch_y = np.array(batch_y)
                    # print("inputs", inputs)
                    yield inputs, batch_y.astype('float32')
                    batch_x = [[], []]
                    batch_y = []
                    # inputs = {}

    # vocab is a map from word to index
    # it can be used to convert words to indexes (in Trainer.vectorize_sequence),
    # which can then be used as input to an embeddings layer

    def encode_sentence(self,s):
        tokens = list(self.tokenizer.tokenize(s))
        tokens = tokens[:50] #padding
        tokens.append(SEP)
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def bert_encode(self,premise, hypothesis, tokenizer):
        
        sentence1 = tf.ragged.constant([self.encode_sentence(s) for s in np.array(premise)])
        sentence2 = tf.ragged.constant([self.encode_sentence(s) for s in np.array(hypothesis)])
        
        clss = [tokenizer.convert_tokens_to_ids([CLS])]*(sentence1.shape[0])
        input_word_ids = tf.concat([clss, sentence1, sentence2], axis=-1)


        input_masks = tf.ones_like(input_word_ids).to_tensor()

        type_cls = tf.zeros_like(clss)
        type_s1 = tf.zeros_like(sentence1)
        type_s2 = tf.ones_like(sentence2)
        input_type_ids = tf.concat([type_cls, type_s1, type_s2], axis=-1).to_tensor()

        inputs = {
          'input_word_ids': input_word_ids.to_tensor(),
          'input_masks': input_masks,
          'input_type_ids': input_type_ids}
      
        return inputs
    
    def create_vocab(self, df):

        sentences = df['premise'].tolist() + df['hypothesis'].tolist()

        # words = [[word for word in sentence.split(' ')] for sentence in sentences]
        word_set = set()
        for sentence in sentences:
            words = self.preprocess_sentence(sentence).split(' ')
            for word in words:
                word_set.add(word)
        word_set.add(UNK)
        word_set.add(PAD)
        word_set.add(START)
        word_set.add(END)

        vocab = {word: index for index, word in enumerate(word_set)}
        return vocab

    # def unicode_to_ascii(self, s):
    #     return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

    def preprocess_sentence(self, w):
        # https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt
        w = w.lower().strip()

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,??])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,??]+", " ", w)

        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = START + ' ' + w + ' ' + END
        return w

    def pad_sequences(self, batch_x, pad_value):
        ''' This function should take a batch of sequences of different lengths
            and pad them with the pad_value token so that they are all the same length.
            batch_x is a list of lists.
        '''
        pad_length = len(max(batch_x, key=lambda x: len(x)))
        for i, x in enumerate(batch_x):
            if len(x) < pad_length:
                batch_x[i] = x + ([pad_value] * (pad_length - len(x)))
        return batch_x

    # returns a list of list of tokens
    def preprocess_sentences(self, sentences_a):
        processed_sentences = []
        for sentence in sentences_a:
            tokenized = self.tokenizer.tokenize(sentence) if self.use_bert else self.preprocess_sentence(
                sentence).split(' ')
            tokenized = tokenized if self.use_bert else [START] + tokenized + [END]

            processed_sentences.append(tokenized)
        return processed_sentences

    def vectorize_sequence(self, seq):
        seq = [tok if tok in self.vocab else UNK for tok in seq]
        return [self.vocab[tok] for tok in seq]

    def unvectorize_sequence(self, seq):
        translate = sorted(self.vocab.keys(), key=lambda k: self.vocab[k])
        return [translate[i] for i in seq]

    def one_hot_encode_label(self, label):
        vec = [1.0 if l == label else 0.0 for l in range(3)]
        return vec

    def run_training_loop(self):
        if self.baseline:
            # TODO implement this below
            self.run_training_loop_approach_one()
            return
        

        
        # must complie the model
        # TODO should the optimizer be passed in as a command line argument?
        # optimizer = tf.optimizers.Adadelta(clipvalue=0.5)
        # self.nli_classifier.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        logdir = "logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S") + f'--{self.params["classifier"]}--do-{self.params["dropout"]}--hs-{self.params["hidden_size"]}--em-{self.params["embedding_size"]}'
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
        
        if self.use_bert: 

            self.nli_classifier.classifier.fit(self.train_batch_generator, validation_data=self.dev_batch_generator, epochs = self.params['epochs'], 
                    batch_size = self.params['batch_size'],callbacks=[tensorboard_callback])


        if self.use_gru:
            self.nli_classifier.classifier.fit(self.train_batch_generator, epochs=self.params['epochs'],
                # steps_per_epoch=self.train_data.shape[0]/self.params['batch_size'],
                                        callbacks=[tensorboard_callback],
                                        batch_size=self.params['batch_size'],
                                        validation_data=self.dev_batch_generator
                               )
        elif not self.use_bert:
            self.nli_classifier.fit(self.train_batch_generator, epochs=self.params['epochs'],
                # steps_per_epoch=self.train_data.shape[0]/self.params['batch_size'],
                                        callbacks=[tensorboard_callback],
                                        batch_size=self.params['batch_size'],
                                        validation_data=self.dev_batch_generator
                                   )
#         for i in range(self.params['epochs']):
#             print(f'Epoch {i+1} / {self.params["epochs"]}')
#             # Training
#             # self.nli_classifier.fit(self.batch_generator_nli(self.train_data, self.train_labels, self.params['batch_size']), epochs=1, 
#             # steps_per_epoch=self.train_data.shape[0]/self.params['batch_size'])
#             self.nli_classifier.fit(self.train_batch_generator, epochs=1, 
#             steps_per_epoch=self.train_data.shape[0]/self.params['batch_size'], 
#                                     callbacks=[tensorboard_callback])
            
#             #Evaluation
#             # loss, acc = self.nli_classifier.evaluate(self.batch_generator_nli(self.dev_data, self.dev_labels),
#             # steps=self.dev_data.shape[0])
#             loss, acc, recall, precision, f1 = self.nli_classifier.evaluate(self.dev_batch_generator,
#             steps=self.dev_data.shape[0], callbacks=[tensorboard_callback])
#             print('Dev Loss:', loss, 'Dev Acc:', acc, 'Dev recall:', recall, 'Dev precision:', precision, 'Dev f1:', f1)

        # If using test set (after finding your best model only)
        # test_loss, test_acc = self.nli_classifier.evaluate(self.batch_generator_nli(self.test_data, self.test_labels),
        # steps=self.test_data.shape[0])

        # uncomment to test final models
        # test_loss, test_acc, test_recall, test_precision, test_f1 = self.nli_classifier.evaluate(self.test_batch_generator,
        # steps=self.test_data.shape[0])
        # print('Test Loss:', test_loss, 'Test Acc:', test_acc, 'Test recall:', test_recall, 'Test precision:', test_precision, 'Test f1:', test_f1)

    # TODO implement this method
    def run_training_loop_approach_one(self):
        import matplotlib.pyplot as plt
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        # Train a random forest classifier
        rf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=3)
        baseline = NLI_Baseline()
        train_fm_data=baseline.fv(self.train_data)
        dev_fm_data=baseline.fv(self.dev_data)
        test_fm_data=baseline.fv(self.test_data)
        rf.fit(train_fm_data, self.train_labels)
        dev_preds = rf.predict(dev_fm_data)
        test_preds = rf.predict(test_fm_data)
        print('baseline dev result', classification_report(self.dev_labels, dev_preds))
        print('baseline test result', classification_report(self.test_labels, test_preds))
        # Feature importances
        features =["BLEU",'overlap_count','polarity','subjectivity','similarity','bert_sim','eucd','jaccd']
        importances = rf.feature_importances_
        indices = np.argsort(importances)
        # Plot the feature importances of the forest
        plt.title('Feature Importances of Random Forest')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()

    # helper function for debugging
    def run_on_example(self):
        premise_ex, hypothesis_ex = self.train_data['premise'].tolist()[0], self.train_data['hypothesis'].tolist()[0]
        premise_ids, hypothesis_ids = self.vectorize_sequence(premise_ex), self.vectorize_sequence(hypothesis_ex)

        inputs = {
            'premise': np.array([premise_ids]),
            'hypothesis': np.array([hypothesis_ids])
        }
        output = self.nli_classifier(inputs)
        print(output)

