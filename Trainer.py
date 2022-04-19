from Classifier import NLI_Classifier

class NLI_Trainer:
    def __init__(self, params):
        self.params = params
        
        self.nli_classifier = NLI_Classifier(self.params['classifier_params'])
        
    def run_training_loop(self):
        pass 