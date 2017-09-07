# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree. 
#

import os
import numpy as np
import nltk
import pickle as pkl


    
def get_batch(batch, word_vec):
    # sent in batch in decreasing order of lengths (bsize, max_len, word_dim)
    lengths = np.array([len(x) for x in batch])
    max_len = np.max(lengths)
    embed = np.zeros((max_len, len(batch), 300))
    
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            embed[j, i, :] = word_vec[batch[i][j]]

    return torch.from_numpy(embed).float(), lengths 


def get_word_dict(sentences, tokenize=True):
    # create vocab of words
    word_dict = {}
    sentences = [s.split() if not tokenize else nltk.word_tokenize(s) for s in sentences]
    for sent in sentences:
        for word in sent:
            if word not in word_dict:
                word_dict[word] = ''
    word_dict['<s>'] = ''
    word_dict['</s>'] = ''
    word_dict['<p>'] = ''
    return word_dict

def get_glove(word_dict, glove_path):
    # create word_vec with glove vectors
    word_vec = {}
    with open(glove_path) as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.array(list(map(float, vec.split())))
    

    print('Found {0}(/{1}) words with glove vectors'.format(len(word_vec), len(word_dict)))
    return word_vec

def get_skipthought(word_dict, glove_path):

    word_vec = {}
    embeddings = np.load(glove_path + 'expanded_embeddings.npy')
    with open(glove_path + 'expanded_vocab.pkl', 'rb') as f:
        expanded_vocab = pkl.load(f)
    for word in expanded_vocab.keys():
        if word in word_dict:
            word_vec[word] = embeddings[expanded_vocab[word]]
    word_vec['<s>'] = np.random.random([620])
    word_vec['</s>'] = np.random.random([620])
    word_vec['<p>'] = np.random.random([620])
    print('Found {0}(/{1}) words with skipthought vectors'.format(len(word_vec), len(word_dict)))
    print(word_vec['<s>'].shape)
    return word_vec


def build_vocab(sentences, glove_path, skipthought = False, tokenize=True):
    word_dict = get_word_dict(sentences, tokenize)
    if skipthought:
        word_vec = get_skipthought(word_dict, glove_path)
    else:
        word_vec = get_glove(word_dict, glove_path)
    print('Vocab size : {0}'.format(len(word_vec)))
    print(word_vec['<s>'].shape)
    return word_vec

def get_nli(data_path):
    s1 = {}
    s2 = {}
    target = {}
    
    dico_label = {'entailment':0,  'neutral':1, 'contradiction':2}
    
    for data_type in ['train', 'dev', 'test']:
        s1[data_type], s2[data_type], target[data_type] = {}, {}, {}
        s1[data_type]['path'] = os.path.join(data_path, 's1.' + data_type)
        s2[data_type]['path'] = os.path.join(data_path, 's2.' + data_type)
        target[data_type]['path'] = os.path.join(data_path, 'labels.' + data_type)
        
        s1[data_type]['sent'] = [line.rstrip() for line in open(s1[data_type]['path'], 'r')]
        s2[data_type]['sent'] = [line.rstrip() for line in open(s2[data_type]['path'], 'r')]
        target[data_type]['data'] = np.array([dico_label[line.rstrip('\n')] for line in open(target[data_type]['path'], 'r')])
        
        assert len(s1[data_type]['sent']) == len(s2[data_type]['sent']) == len(target[data_type]['data'])
        
        print('** {0} DATA : Found {1} pairs of {2} sentences.'.format(
                            data_type.upper(), len(s1[data_type]['sent']), data_type))
        
        
    train = {'s1':s1['train']['sent'], 's2':s2['train']['sent'], 'label':target['train']['data']}
    dev = {'s1':s1['dev']['sent'], 's2':s2['dev']['sent'], 'label':target['dev']['data']}
    test  = {'s1':s1['test']['sent'] , 's2':s2['test']['sent'] , 'label':target['test']['data'] }
    return train, dev, test

def txt_to_sent(sentences, word_vec, tokenize=True):

    sentences = [['<s>']+s.split()+['</s>'] if not tokenize else ['<s>']+nltk.word_tokenize(s)+['</s>'] for s in sentences]
    n_w = np.sum([len(x) for x in sentences])
    
    # filters words without glove vectors
    for i in range(len(sentences)):
        s_f = [word for word in sentences[i] if word in word_vec]
        if not s_f:
            import warnings
            warnings.warn('No words in "{0}" (idx={1}) have glove vectors. Replacing by "</s>"..'.format(sentences[i], i))
            s_f = ['</s>']
        sentences[i] = s_f

    lengths = np.array([len(s) for s in sentences])
    n_wk = np.sum(lengths)

    print('Nb words kept : {0}/{1} ({2} %)'.format(n_wk, n_w, round((100.0 * n_wk) / n_w, 2)))
                                              
    return sentences


