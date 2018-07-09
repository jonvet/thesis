import numpy as np
import pandas as pd
import pickle as pkl
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infersent.data import get_nli

MODEL_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/models/toronto_n5/'
SNLI_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/dataset/SNLI/'
SAVE_PATH = './'

with open(MODEL_PATH + 'vocab.pkl', 'rb') as f:
    vocab = pkl.load(f)

dictionary = defaultdict(int)

print('Loading corpus')
train, dev, test = get_nli(SNLI_PATH)
train = np.array(train['s2'])
dev = np.array(dev['s2'])
test = np.array(test['s2'])

for part in [train, dev, test]:
    for i in range(len(part)):
        sentence = part[i].split()
        for word in sentence:
            dictionary[word] += 1

def get_count_df(dic):
    df = pd.DataFrame({'word': [k for k,v in dic.items()],
        'count': [v for k,v in dic.items()]})
    df = df.sort_values(by='count', ascending=False)
    df['cumsum'] = np.cumsum(df['count'].values)
    df['cumsum_rel'] = df['cumsum'] / df['count'].sum()
    return df

full_df = get_count_df(dictionary)
short_dict = {k:v for k,v in zip(full_df['word'][:500], np.arange(500))}

adjectives = list(wn.all_synsets(wn.ADJ))
adjectives = [a.name().split('.')[0] for a in adjectives]

adj = [a for a in dictionary.keys() if a in adjectives]
adj_dictionary = {k:dictionary[k] for k in adj}
adj_df = get_count_df(adj_dictionary)

sent = SentimentIntensityAnalyzer()
adj_df['pos'] = adj_df['word'].apply(lambda x: sent.polarity_scores(x)['pos'])
adj_df['neg'] = adj_df['word'].apply(lambda x: sent.polarity_scores(x)['neg'])
adj_df['compound'] = adj_df['word'].apply(lambda x: sent.polarity_scores(x)['compound'])
print(adj_df.loc[(adj_df['compound']>0.1) & (adj_df['compound']<0.2)])

adj_dictionary_pos = {k:v for k,v in adj_df.loc[(adj_df['compound']>0.3)][['word', 'count']].values}
adj_df_pos = get_count_df(adj_dictionary_pos)
adj_dictionary_neg = {k:v for k,v in adj_df.loc[(adj_df['compound']<-0.3)][['word', 'count']].values}
adj_df_neg = get_count_df(adj_dictionary_neg)


cr_train = pd.read_csv('/Users/Jonas/Documents/Repositories/SentEval/data/senteval_data/CR/binary_nlp_cr_train.csv')
cr_test = pd.read_csv('/Users/Jonas/Documents/Repositories/SentEval/data/senteval_data/CR/binary_nlp_cr_test.csv')
cr = pd.concat((cr_train, cr_test))
pos_count = 0
neg_count = 0
positives = adj_df_pos['word'].values[:100]
negatives = adj_df_neg['word'].values[:100]
for i, sent in enumerate(cr['text'].values):
    print(f'\r{i}/{len(cr)}', end='     ')
    s = sent.lower().split()
    for a in positives:
        if a in s:
            pos_count += 1
    for a in negatives:
        if a in s:
            neg_count += 1
            break
adj_dic = {w:i for i,w in enumerate(np.concatenate((adj_df_pos['word'].values[:100], adj_df_neg['word'].values[:100]),0))}
with open('adj_dic.pkl', 'wb') as f:
    pkl.dump(adj_dic, f)
    
one_hot = np.zeros([len(cr), len(adj_dic)])
for e,sent in enumerate(cr['text'].values):
    ints = []
    for w in sent.lower().split():
        if w in adj_dic.keys():
            ints.append(adj_dic[w])
    for i in ints:
        one_hot[e,i] = 1 

