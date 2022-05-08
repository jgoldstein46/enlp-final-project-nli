from nltk.grammar import CFG
from nltk.parse.corenlp import CoreNLPParser, CoreNLPServer
from DataFrame import DataFrame
import pandas as pd
import pickle

def main(): 
    # grammar = CFG().chomsky_normal_form()
    # parser = ViterbiParser(grammar)
    # parser = GenericStanfordParser(
    #     '../stanford-parser-full-2020-11-17/stanford-parser.jar',
    #     '../stanford-parser-full-2020-11-17/stanford-parser-4.2.0-models.jar'
    # )
    df = DataFrame(pd.read_csv('train.csv'))
    df.create_english()
    df = df.english_df
    
    premise = df['premise'].tolist()
    hypothesis = df['hypothesis'].tolist()
    # prem_hyp = premise + hypothesis 
#    df['premise_']
    server = CoreNLPServer('../stanford-corenlp-4.4.0/stanford-corenlp-4.4.0.jar', '../stanford-corenlp-4.4.0/stanford-corenlp-4.4.0-models.jar')
    
    server.start()
    parser = CoreNLPParser()
    # print(parser.parse_sents(['This is a sample sentence to parse']))
    # sent_generator = parser.raw_parse_sents(['This is a sample sentence to parse', 'This is a second sentence'])
    # print(hypothesis[0:5])
    # print(hypothesis[370:375])
    sent_generator = parser.raw_parse_sents(hypothesis)
    hypothesis_trees = [None for _ in range(len(hypothesis))]
    j = 0
    for generator in sent_generator:
        for sent in generator: 
            hypothesis_trees[j] = pickle.dumps(sent)
            j += 1 
            
    df['hypothesis_tree'] = hypothesis_trees
    print(pickle.loads(df['hypothesis_tree'].tolist()[0]))
    # s_pickled = None
    # print (sent_generator.next())
    # premises = 
    # for sent in sent_generator:
    #     for s in sent: 
    #         # print(s)
            # s_pickled = pickle.dumps(s)
    server.stop()
    # s_loaded = pickle.loads(s_pickled)
    # print(s_loaded)
    # sent_generator_s = pickle.dumps(sent_generator)
    # sent_generator_loaded = pickle.loads(sent_generator_s)
    # for sent in sent_generator_loaded:
    #     for s in sent: 
    #         print(s)

if __name__ == '__main__':
    main()
    