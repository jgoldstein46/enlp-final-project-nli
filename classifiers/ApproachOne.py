import pandas as pd
from itertools import product
import numpy as np
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from math import sqrt, pow, exp
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from classifiers.Classifier import NLI_Classifier_Base
from DataFrame import DataFrame

class NLI_Baseline(NLI_Classifier_Base):
    def __init__(self, params):
        self.params = params
        self.train_file = params['train']
        df = DataFrame(pd.read_csv(self.train_file))
        # start out with only english 
        data = df.english_df
        
        
    def bleu (self, data):
        '''The BLEU score of the hypothesis with respect to the premise, using an n-gram length between 1 and 4'''
        '''data: data parition (train/dev/test)'''
        '''return a list of BLEU score'''
        b = data.apply(lambda row: sentence_bleu(row['premise'],row['hypothesis'], weights=(0.25, 0.25, 0.25, 0.25)), axis=1)
        bleu=b.tolist()
        return bleu
    
    def normalize(self, comment, lowercase, remove_stopwords):
        '''preprocessing: lowercase, remove stopwords, lemmatize'''
        nlp = spacy.load('en_core_web_sm')
        stops = nlp.Defaults.stop_words
        if lowercase:
            comment = comment.lower()
        comment = nlp(comment)
        lemmatized = list()
        for word in comment:
            lemma = word.lemma_.strip()
            if lemma:
                if not remove_stopwords or (remove_stopwords and lemma not in stops):
                    lemmatized.append(lemma)
        return " ".join(lemmatized)
    
    def euclidean_distance(self, x,y):
        ''' return euclidean distance between two lists of sentences'''

        return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))

    def distance_to_similarity(self, distance):
        return 1/exp(distance)
    
    def jaccard_similarity(self, x, y):
        ''' returns the jaccard similarity between two lists of sentences'''
        intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
        union_cardinality = len(set.union(*[set(x), set(y)]))
        return intersection_cardinality/float(union_cardinality)

    def bert_cos_sim (self, p,h):
        '''Take 2 lists of string, return a list of cosine similarity between two lists using SentenceTransformer'''
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        pm=model.encode(p, convert_to_tensor=True)
        hm=model.encode(h, convert_to_tensor=True)
        cosine_scores = util.cos_sim(pm, hm)
        bert_sim=np.diag(cosine_scores)
        bert_sim=bert_sim.tolist()
        return bert_sim
    
    def word_overlap (self, data):
        '''
        The number of words that occur both in hypothesis and premise
        '''
        data['word_overlap'] = data.apply(lambda r: set(r['premise'].lower().split()) & set(r['hypothesis'].lower().split()),axis=1)

        return data['word_overlap'].str.len()
    
    def sent_polarity(self, p, h):
        '''take a list of sentence as input, return a list of polarity score and the sum of polarity score between premise and hypothesis'''

        sent_p=[]
        sent_h=[]
        for i in p:
            res=TextBlob(i).sentiment.polarity
            sent_p.append(res)

        for j in h:
            res=TextBlob(j).sentiment.polarity
            sent_h.append(res)

        sum_list = [a + b for a, b in zip(sent_p, sent_h)]

        return sent_p, sent_h, sum_list
    
    def subj(self, p, h):
        '''take a list of sentence as input, return a list of subjectivity score and the difference of subjectivity score between premise and hypothesis'''

        subj_p=[]
        subj_h=[]
        for i in p:
            res=TextBlob(i).sentiment.subjectivity
            subj_p.append(res)

        for j in h:
            res=TextBlob(j).sentiment.subjectivity
            subj_h.append(res)

        diff_list = [a - b for a, b in zip(subj_p, subj_h)]

        return subj_p, subj_h, diff_list
    
    def w2v_cos_sim(self, p,h):
        '''take a list of string as input, output a list of word2vec cosine similarity score between premise and hypothesis'''
        nlpm = spacy.load("en_core_web_md")
        cos_sim=[]
        for i,v in zip(p,h):
            p=nlpm(i)
            h=nlpm(v)
            score=p.similarity(h)
            cos_sim.append(score)
        return cos_sim
    
    
