from Trainer import NLI_Trainer
from classifiers import *
from classifiers.ApproachOne import NLI_Baseline
from classifiers.ApproachThree import LSTM_NLI_Classifier
from classifiers.EISM import EISM
from classifiers.ApproachFour import GRU_NLI_Classifier
from classifiers.ApproachTwo import BERT_NLI_Classifier
import tensorflow as tf
import pandas as pd
from classifiers.GPT2Wrapper import GPT2Wrapper
from classifiers.EISMModified import ModifiedEISM

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=32) # batch size before weights update
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    # parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('-do','--dropout', default=0.5, type=float, help='Dropout rate')
    parser.add_argument('-em','--embedding-size', default=300, type=int, help='Embedding dimension size')
    parser.add_argument('-hs','--hidden-size', default=300, type=int, help='Hidden layer size')
    # parser.add_argument('--use-bert', action='store_true', help='Use BERT tokenization and embeddings')
    # parser.add_argument('-T','--train', type=str, help='Train file', required=True)
    # parser.add_argument('-t','--test', type=str, help='Test or Dev file', required=True)
    # parser.add_argument('-b','--batch-size', default=10, type=int, help='Batch Size')
    # parser.set_defaults(use_bert=False)
    parser.add_argument('-d','--device', default='cpu', help='Either "cpu" or "cuda"')
    parser.add_argument('-e','--epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('-s', '--seed', default=100, type=int, help='Train, dev, test split random seed number')
    parser.add_argument('-t', '--train', default='train.csv', type=str, help='file to be used for training and dev and testing after split')
    parser.add_argument('-c', '--classifier', default='baseline', type=str, help="""The name of the classifier you wish to use. 
    baseline: Baseline feature set, bert: BERT with LSTM model, lstm: Glove with LSTM model, gru: GloVe with GRU, eism: EISM model, 
    eism-m: Our Modified EISM model.""")
    parser.add_argument('-g', '--glove-file', help='path to glove embeddings file', type=str, default='../../data/glove.840B.300d.txt')


    args = parser.parse_args()

    # convert to dict
    params = vars(args)
    # TODO put your classes here to be used
    classifier2class = {'baseline': NLI_Baseline,
    'bert': BERT_NLI_Classifier,
    'lstm': LSTM_NLI_Classifier,
    'eism': EISM,
    'gru': GRU_NLI_Classifier, 
    'gpt2': GPT2Wrapper,
    'eism-m': ModifiedEISM
    }
    
    params['nli_classifier_class'] = classifier2class[params['classifier']]
    params['classifier_params'] = {
        'n_layers': params['n_layers'],
        'dropout': params['dropout'],
        'hidden_size': params['hidden_size'],
        'batch_size': params['batch_size'],
        'embedding_size': params['embedding_size'],
        'glove_file': params['glove_file'],
        'train': params['train'],
        'classifier' : params['classifier']
    }
    # tf.config.experimental.list_physical_devices('GPU')
    trainer = NLI_Trainer(params)
    # pd.set_option('display.max_colwidth', None)
    # pd.set_option('display.max_columns', 10)
    # print(trainer.train_data.head(), trainer.train_labels.head(), sep='\n')
    trainer.run_training_loop()
    # trainer.run_on_example()
    
    # print("Finished running the training loop")
    
if __name__ == '__main__': 
    main()
