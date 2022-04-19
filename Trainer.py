from classifiers.Classifier import NLI_Classifier

class NLI_Trainer:
    def __init__(self, params):
        self.params = params
        classifier = self.params['classifier_class']
        self.nli_classifier = classifier(self.params['classifier_params'])

    def run_training_loop(self):
        pass 