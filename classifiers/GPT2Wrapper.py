from transformers import GPT2Tokenizer, TFGPT2ForSequenceClassification
import tensorflow as tf
from keras import Model
from classifiers.Classifier import NLI_Classifier_Base

class GPT2Wrapper(NLI_Classifier_Base):
    def __init__(self, params):
        super().__init__(params)
        self.GPT = TFGPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=3, pad_token_id = 50256)
        # self.GPT.pad_token = params['tokenizer'].pad_token 
        # self.tokenizer = GPT2Tokenizer.from_pretrained("microsoft/DialogRPT-updown")
        self.compile()
    def call(self, inputs):
#         Get the inputs using the tokenizer correctly 
# then you pass it into gpt 2 . Maybe you will have to use 
# the vocabulary for this to get back to tokens. Or redisgn the code
# for trainer. IDK 
        # inputs = tokenizer()
        output = self.GPT(**inputs)
        print(output.logits.shape)
        
        return output.logits