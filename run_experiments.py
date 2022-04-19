from Trainer import NLI_Trainer
from classifiers import *
from classifiers.ApproachOne import NLI_Baseline

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '-b', type=int, default=50) # batch size before weights update
    parser.add_argument('--n_layers', '-l', type=int, default=2)
    # parser.add_argument('--size', '-s', type=int, default=64)
    parser.add_argument('-do','--dropout', default=0.3, type=float, help='Dropout rate')
    parser.add_argument('-em','--embedding-size', default=100, type=int, help='Embedding dimension size')
    parser.add_argument('-hs','--hidden-size', default=10, type=int, help='Hidden layer size')
    # parser.add_argument('--use-bert', action='store_true', help='Use BERT tokenization and embeddings')
    # parser.add_argument('-T','--train', type=str, help='Train file', required=True)
    # parser.add_argument('-t','--test', type=str, help='Test or Dev file', required=True)
    parser.add_argument('-b','--batch-size', default=10, type=int, help='Batch Size')
    # parser.set_defaults(use_bert=False)
    parser.add_argument('-d','--device', default='cpu', help='Either "cpu" or "cuda"')
    parser.add_argument('-e','--epochs', default=20, type=int, help='Number of epochs')
    parser.add_argument('-s', '--seed', default=100, type=int, help='Train, dev, test split random seed number')
    parser.add_argument('-t', '--train', default='train.csv', type='string', help='file to be used for training and dev and testing after split')
    parser.add_argument('-c', '--classifier', default='baseline', type=str, help="""The name of the classifier you wish to use. 
    baseline --> approach one, bert --> approach two, lstm --> approach three, gru --> approach four""")
    parser.add_argument('-g', '--glove-file', help='path to glove embeddings file', default=None)


    args = parser.parse_args()

    # convert to dict
    params = vars(args)
    # TODO put your classes here to be used
    classifier2class = {'baseline': NLI_Baseline,
    'bert': None,
    'lstm': None,
    'gru': None,
    }
    params['nli_classifier_class'] = classifier2class[params['classifier']]
    params['classifier_params'] = {
        'n_layers': params['n_layers'],
        'dropout': params['dropout'],
        'hidden-size': params['hidden-size'],
        'batch-size': params['batch-size'],
        'embedding-size': params['embedding-size'],
        'glove-file': params['glove-file']
    }

    trainer = NLI_Trainer(params)
    trainer.run_training_loop()
