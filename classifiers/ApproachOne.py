import pandas as pd
from itertools import product
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
from math import sqrt, pow, exp
from sentence_transformers import SentenceTransformer, util
from classifiers.Classifier import NLI_Classifier_Base
from DataFrame import DataFrame

class NLI_Baseline():
    def __init__(self, data):
        df = DataFrame(pd.read_csv(self.train_file))
        # start out with english data
        self.vectorizer = CountVectorizer(stop_words='english')
        data = df.english_df #data is a df


    def CV_base(self, data):
        """
            Reads a list of string
            returns CountVectorizer matrix for every instance of premise/hypothesis
        """
        from sklearn.feature_extraction.text import CountVectorizer
        p=data['premise'].tolist() # a list of sentence (string)
        h=data['hypothesis'].tolist()
        p_word_count=self.vectorizer.fit_transform(p)
        h_word_count=self.vectorizer.fit_transform(h)
        parr= np.array([p]) #2d array
        harr=np.array([h])
        parr=parr.T
        harr=harr.T
        self.cv_base=np.concatenate((parr, harr), axis=1)
        return self.cv_base

    def bleu (self, data):
        '''The BLEU score of the hypothesis with respect to the premise, using an n-gram length between 1 and 4'''
        '''data: data parition (train/dev/test)'''
        '''return a list of BLEU score'''
        from nltk.translate.bleu_score import sentence_bleu
        b = data.apply(lambda row: sentence_bleu(row['premise'],row['hypothesis'], weights=(0.25, 0.25, 0.25, 0.25)), axis=1)
        bleu=b.tolist()
        bleu=np.array([bleu])
        bleu_arr=bleu.T
        return bleu_arr

    def normalize(self, comment, lowercase, remove_stopwords):
        '''
        preprocessing function: lowercase, remove stopwords, lemmatize
        return clean list of sentence (string)
        '''
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
        return " ".join(lemmatized) # a string

    def clean(self, data):
        '''
        use the preprocessing function above
        return clean list of sentence (string)
        '''
        p_clean=data['premise'].apply(self.normalize, lowercase=True, remove_stopwords=True)
        h_clean=data['hypothesis'].apply(self.normalize, lowercase=True, remove_stopwords=True)
        p_clean=p_clean.values.tolist()
        h_clean=h_clean.values.tolist()
        return p_clean, h_clean

    def euclidean_distance(self, data):
        '''
        Takes two clean list of sentences as input
        return euclidean distance between two lists of sentences
        '''
        nlp = spacy.load('en_core_web_sm')
        p_clean, h_clean=self.clean(data)
        output = list (map (lambda x,y: [x,y],p_clean,h_clean))
        euc=[]
        for p in output:
            embeddings = [nlp(pair).vector for pair in p]
            e=sqrt(sum(pow(a-b,2) for a, b in zip(embeddings[0], embeddings[1])))
            dist=1/exp(e)
            euc.append(dist)
        euc=np.array([euc])
        euc=euc.T
        return euc

    def jaccard_similarity(self, data):
        ''' returns the jaccard similarity between two lists of sentences'''
        p_clean, h_clean=self.clean(data)
        output = list (map (lambda x,y: [x,y],p_clean,h_clean))
        jd=[]
        for p in output:
            sentences = [pair.split(" ") for pair in p]
            intersection_cardinality = len(set.intersection(*[set(sentences[0]), set(sentences[1])]))
            union_cardinality = len(set.union(*[set(sentences[0]), set(sentences[1])]))
            jdd=intersection_cardinality/float(union_cardinality)
            jd.append(jdd)
        jd=np.array([jd])
        jd=jd.T
        return jd

    def bert_cos_sim (self, data):
        '''
        Take 2 lists of string, return a list of cosine similarity between two lists using SentenceTransformer
        '''
        from sklearn.metrics.pairwise import cosine_similarity
        p=data['premise'].tolist() # a list of sentence (string)
        h=data['hypothesis'].tolist()
        model = SentenceTransformer('bert-base-nli-mean-tokens')
        pm=model.encode(p, convert_to_tensor=True)
        hm=model.encode(h, convert_to_tensor=True)
        cosine_scores = util.cos_sim(pm, hm)
        bert_sim=np.diag(cosine_scores)
        bert_sim=bert_sim.tolist()
        bert_sim=np.array([bert_sim])
        bert_sim=bert_sim.T
        return bert_sim

    def word_overlap (self, data):
        '''
        The number of words that occur both in hypothesis and premise
        '''
        data['word_overlap'] = data.apply(lambda r: set(r['premise'].lower().split()) & set(r['hypothesis'].lower().split()),axis=1)
        wo=data['word_overlap'].str.len().tolist()
        wo=np.array([wo])
        wo=wo.T
        return wo

    def sent_polarity(self, data):
        '''take a list of sentence as input, return a list of polarity score and the sum of polarity score between premise and hypothesis'''
        from textblob import TextBlob
        p_clean, h_clean=self.clean(data)
        sent_p=[]
        sent_h=[]
        for i in p_clean:
            res=TextBlob(i).sentiment.polarity
            sent_p.append(res)

        for j in h_clean:
            res=TextBlob(j).sentiment.polarity
            sent_h.append(res)

        sum_list = [a + b for a, b in zip(sent_p, sent_h)]
        sum_list=np.array([sum_list])
        sum_list=sum_list.T
        return sum_list

    def subj(self, data):
        '''take a list of sentence as input, return a list of subjectivity score and the difference of subjectivity score between premise and hypothesis'''
        from textblob import TextBlob
        p_clean, h_clean=self.clean(data)
        subj_p=[]
        subj_h=[]
        for i in p_clean:
            res=TextBlob(i).sentiment.subjectivity
            subj_p.append(res)

        for j in h_clean:
            res=TextBlob(j).sentiment.subjectivity
            subj_h.append(res)

        diff_list = [a - b for a, b in zip(subj_p, subj_h)]
        diff_list=np.array([diff_list])
        diff_list=diff_list.T
        return diff_list

    def w2v_cos_sim(self, data):
        '''take a list of string as input, output a list of word2vec cosine similarity score between premise and hypothesis'''
        nlpm = spacy.load("en_core_web_md")
        p_clean, h_clean=self.clean(data)
        cos_sim=[]
        for i,v in zip(p_clean,h_clean):
            p=nlpm(i)
            h=nlpm(v)
            score=p.similarity(h)
            cos_sim.append(score)
        cos_sim=np.array([cos_sim])
        cos_sim=cos_sim.T
        return cos_sim

    def fv(self,data):
        baseline=NLI_Baseline(data)
        #cv_base=baseline.CV_base(data) #remove cv_base gets 1% inprovement in accuracy
        bleu_m=baseline.bleu(data)
        eud_m=baseline.euclidean_distance(data)
        judm=baseline.jaccard_similarity(data)
        besim=baseline.bert_cos_sim(data)
        wom=baseline.word_overlap(data)
        plm=baseline.sent_polarity(data)
        sbm=baseline.subj(data)
        w2vm=baseline.w2v_cos_sim(data)
        fm=np.concatenate((bleu_m,eud_m,judm,besim,wom,plm,sbm,w2vm), axis=1) #remove cv_base gets 1% inprovement in accuracy
        return fm





    
