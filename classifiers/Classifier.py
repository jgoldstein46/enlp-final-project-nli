# base class for classifiers 
import numpy as np

class NLI_Classifier_Base():
    

    def __init__(self, params):
        self.glove_file = params['glove-file']
        self.vocab = params['vocab']
        self.embedding_size = params['embedding-size']
        self.embeddings_matrix = self.load_pretrained_embeddings()
        self.model = None

    def load_pretrained_embeddings(self):
        # TODO override this function in subclasses if necessary 
        raise NotImplementedError
    
    
    