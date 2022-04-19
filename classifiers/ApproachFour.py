from Classifier import NLI_Classifier_Base
import numpy as np

class GRU_NLI_Classifier(NLI_Classifier_Base):
    def __init__(self, params):

        pass
    def load_pretrained_embeddings(self):
        embedding_matrix = np.zeros((len(self.vocab), self.embedding_size))
        with open(self.glove_file, encoding='utf8') as f:
            for line in f:
                # Each line will be a word and a list of floats, separated by spaces.
                # If the word is in your vocabulary, create a numpy array from the list of floats.
                # Assign the array to the correct row of embedding_matrix.
                split_line = line.strip().split(" ")
                word = split_line[0]
                if word in self.vocab:
                    embeddings = np.array(split_line[1:])
                    embedding_matrix[self.vocab[word]] = embeddings

        embedding_matrix[self.vocab['[UNK]']] = np.random.randn(self.embedding_size)
        return embedding_matrix