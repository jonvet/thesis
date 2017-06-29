import nltk
import tensorflow as tf
import numpy as np
import operator
import pickle as pkl
from collections import defaultdict
import os
import glob

def word_vocab(path, vocab_name=None, vocab=None):

    if vocab == None:
        vocab = defaultdict(int)
    count = 1
    with open(path, 'r') as f:
        for sentence in f:
            print('\rSentence %d' %count, end='')
            count += 1
            this_sentence = sentence.split()
            for word in this_sentence:
                vocab[word] += 1
    return vocab

def finalise_vocab(raw_vocab, vocab_size):

    '''
    Takes as input a dictionary of unique words that occur in a corpus.
    Returns a sorted dictionary of size vocab_size with unique values.
    '''

    sorted_vocab = sorted(raw_vocab.items(), key=operator.itemgetter(1))
    sorted_vocab = sorted_vocab[-vocab_size:]
    final_vocab = {'<PAD>': 0, '<OOV>': 1, '<GO>': 2, '<END>': 3}
    for word in sorted_vocab:
        final_vocab[word[0]] = len(final_vocab)
    return final_vocab

def txt_to_row(path = './corpus/gingerbread_corpus/rows/'):

    '''
    Takes as input a text file and outputs a text file with each sentence on a separate row
    '''

    temp_parts = glob.glob(path + '*.txt')
    for part in temp_parts:
        with open(part, 'r') as f:
            data = f.read().replace('\n', ' ')
        sentences = nltk.sent_tokenize(data)
        with open(part[:-4] + '_rows.txt', 'w') as f:
            for sentence in sentences:
                f.write('%s\n' % sentence.lower())


def txt_to_sent(path):

    '''
    Imports and tokenises a corpus.
    Returns a list of sentences, where each sentence is itself a list of words
    '''

    sentences = []
    tokenised_sentences = []
    count = 1
    with open(path, 'r') as f:
        for sentence in f:
            print('\rSentence %d' %count, end='')
            this_sentence = sentence.split()
            # this_sentence = nltk.word_tokenize(sentence)
            tokenised_sentences.append(this_sentence)
            count +=1
    return tokenised_sentences

    # with open(path, 'r') as f:
    #     data = f.read().replace('\n', ' ')
    # sentences = nltk.sent_tokenize(data)
    # for sentence in sentences:
    #     this_sentence = nltk.word_tokenize(sentence)
    #     tokenised_sentences.append(this_sentence)
    # return tokenised_sentences

def sent_to_int(path, dictionary, max_sent_len, decoder=False):

    '''
    Encodes sentences using the vocabulary provided
    If decoder = True, it will also return every sentences with a <GO> token in front, and every sentence with a <END> token at the end.
    '''

    lines = 0
    with open(path, 'r') as f:
        for line in f:
            lines += 1
    print('\n%d lines to do\n' % lines)

    if decoder:
        enc_sentences = np.full([lines, max_sent_len], dictionary['<PAD>'], dtype=np.int32)
        dec_go_sentences = np.full([lines, max_sent_len + 2], dictionary['<PAD>'], dtype=np.int32)
        dec_end_sentences = np.full([lines, max_sent_len + 2], dictionary['<PAD>'], dtype=np.int32)
        sentence_lengths = np.full([lines], 0, dtype=np.int32)
    else:
        enc_sentences = np.full([lines, max_sent_len], dictionary['<PAD>'], dtype=np.int32)
        sentence_lengths = []

    i = 0
    if decoder:
        with open(path, 'r') as f:
            for sent in f:
                print('\rSentence %d' %i, end='')
                sentence = sent.split()
                words = []
                for word in sentence:
                    if word not in dictionary:
                        token_id = dictionary['<OOV>']
                    else:
                        token_id = dictionary[word]
                    words.append(token_id)
                sentence_lengths[i] = min(len(sentence),max_sent_len)
                enc_sentences[i, 0:min(len(sentence),max_sent_len)] = words[:min(len(sentence),max_sent_len)]
                dec_go_sentences[i, 0] = dictionary['<GO>']
                dec_go_sentences[i, 1:min(len(sentence),max_sent_len)+1] = words[:min(len(sentence),max_sent_len)]
                dec_go_sentences[i, min(len(sentence),max_sent_len)+1] = dictionary['<END>']
                dec_end_sentences[i, 0:min(len(sentence),max_sent_len)] = words[:min(len(sentence),max_sent_len)]
                dec_end_sentences[i, min(len(sentence),max_sent_len)] = dictionary['<END>']
                i += 1
        return sentence_lengths, max_sent_len + 2, enc_sentences, dec_go_sentences, dec_end_sentences
    
    else:
        with open(path, 'r') as f:
            for sent in f:
                print('\rSentence %d' %i, end='')
                sentence = sent.split()
                words = []
                for word in sentence:
                    if word not in dictionary:
                        token_id = dictionary['<OOV>']
                    else:
                        token_id = dictionary[word]
                    words.append(token_id)
                sentence_lengths[i] = min(len(sentence),max_sent_len)
                enc_sentences[i, 0:min(len(sentence),max_sent_len)] = words[:min(len(sentence),max_sent_len)]
                i += 1
        return sentence_lengths, max_sent_len, enc_sentences, dec_go_sentences, dec_end_sentences


def sick_encode(sentences, dictionary, embeddings = None):

    '''
    Encodes sentences using the vocabulary provided
    '''

    lines = len(sentences)
    max_sent_len = -1
    tokenised_sentences = []
    for sentence in sentences:
        this_sentence = nltk.word_tokenize(sentence)
        tokenised_sentences.append(this_sentence)
        length = len(this_sentence)
        max_sent_len = length if length > max_sent_len else max_sent_len
    print('\n%d lines to do\n' % lines)

    i = 0
    if type(embeddings) == np.ndarray:
        dim = len(embeddings[0])
        enc_sentences = np.full(
            [lines, max_sent_len, dim], dictionary['<PAD>'], 
            dtype=np.float32)
        sentence_lengths = np.full(
            [lines], 0, 
            dtype=np.int32)
        for sentence in tokenised_sentences:
            print('\rSentence %d' %(i+1), end='')
            words = []
            for word in sentence:
                if word not in dictionary:
                    token_id = embeddings[dictionary['<OOV>'],:]
                else:
                    token_id = embeddings[dictionary[word],:]
                words.append(token_id)
            sentence_lengths[i] = min(len(sentence),max_sent_len)
            enc_sentences[i, :min(len(sentence),max_sent_len), :] = np.array(words)[:min(len(sentence),max_sent_len), :]
            i += 1
    else:
        enc_sentences = np.full(
            [lines, max_sent_len], dictionary['<PAD>'], 
            dtype=np.int32)
        sentence_lengths = np.full(
            [lines], 0, 
            dtype=np.int32)
        for sentence in tokenised_sentences:
            print('\rSentence %d' %(i+1), end='')
            words = []
            for word in sentence:
                if word not in dictionary:
                    token_id = dictionary['<OOV>']
                else:
                    token_id = dictionary[word]
                words.append(token_id)
            sentence_lengths[i] = min(len(sentence),max_sent_len)
            enc_sentences[i, 0:min(len(sentence),max_sent_len)] = words[:min(len(sentence),max_sent_len)]
            i += 1
    
    return sentence_lengths, enc_sentences







