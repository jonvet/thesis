import nltk
import tensorflow as tf
import numpy as np
import operator
import pickle as pkl

class word_vocab(object):

    '''
    Takes as input a list of tokenised sentences (where each sentences is a list of words)
    Creates a dictionary, reverse dictionary (both dictionaries) and a sorted dictionary (a list)
    '''

    def __init__(self, tokenised_sentences, corpus_name=None):
        self.dictionary = {'<PAD>': 0, '<OOV>': 1, '<GO>': 2, '<END>': 3}
        for sentence in tokenised_sentences:
            words = []
            for word in sentence:
                if word not in self.dictionary:
                    self.dictionary[word] = len(self.dictionary)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        self.sorted_dictionary = sorted(self.dictionary.items(), key=operator.itemgetter(1))
        self.corpus_name = corpus_name

    def save(self, path):
        with open(path + 'vocab.pkl', 'wb') as f:
            pkl.dump(self, f)

def txt_to_sent(path):

    '''
    Imports data and tokenises data.
    Outputs a list of sentences, where each sentence is itself a list of words
    '''

    sentences = []
    tokenised_sentences = []
    with open(path, 'r') as f:
        data = f.read().replace('\n', ' ')
    sentences = nltk.sent_tokenize(data)
    for sentence in sentences:
        this_sentence = nltk.word_tokenize(sentence)
        tokenised_sentences.append(this_sentence)
    return tokenised_sentences

def sent_to_int(sentences, dictionary, decoder=False, max_sent_len_=None):

    '''
    Encodes sentences using the vocabulary provided
    If decoder = True, it will also return every sentences with a <GO> token in front, and every sentence with a <END> token at the end.
    '''

    data_sentences = []
    max_sent_len = -1
    for sentence in sentences:
        words = []
        for word in sentence:
            if word not in dictionary:
                token_id = dictionary['<OOV>']
            else:
                token_id = dictionary[word]
            words.append(token_id)
        if len(words) > max_sent_len:
            max_sent_len = len(words)
        data_sentences.append(words)
    if max_sent_len_ is not None:
        max_sent_len = max_sent_len_
    if decoder:
        enc_sentences = np.full([len(data_sentences), max_sent_len], dictionary['<PAD>'], dtype=np.int32)
        dec_go_sentences = np.full([len(data_sentences), max_sent_len + 2], dictionary['<PAD>'], dtype=np.int32)
        dec_end_sentences = np.full([len(data_sentences), max_sent_len + 2], dictionary['<PAD>'], dtype=np.int32)
        sentence_lengths = []
        for i, sentence in enumerate(data_sentences):
            sentence_lengths.append(len(sentence))
            enc_sentences[i, 0:len(sentence)] = sentence
            dec_go_sentences[i, 0] = dictionary['<GO>']
            dec_go_sentences[i, 1:len(sentence)+1] = sentence
            dec_go_sentences[i, len(sentence)+1] = dictionary['<END>']
            dec_end_sentences[i, 0:len(sentence)] = sentence
            dec_end_sentences[i, len(sentence)] = dictionary['<END>']
        sentence_lengths = np.array(sentence_lengths, dtype=np.int32)
        return sentence_lengths, max_sent_len + 2, enc_sentences, dec_go_sentences, dec_end_sentences
    else:
        enc_sentences = np.full([len(data_sentences), max_sent_len], dictionary['<PAD>'], dtype=np.int32)
        sentence_lengths = []
        for i, sentence in enumerate(data_sentences):
            sentence_lengths.append(len(sentence))
            enc_sentences[i, 0:len(sentence)] = sentence
        sentence_lengths = np.array(sentence_lengths, dtype=np.int32)
        return sentence_lengths, max_sent_len, enc_sentences

def build_char_dictionary(text, vocab=None, max_sent_len_=None, max_word_len_=None):

    list_of_sentences = nltk.sent_tokenize(text)
    list_of_list_of_words = [] 
    for sentence in list_of_sentences:
        this_sentence = nltk.word_tokenize(sentence)
        list_of_list_of_words.append(this_sentence)

    is_ext_vocab = True
    if vocab is None:
        is_ext_vocab = False
        vocab = {'<PAD>': 0, '<OOV>': 1, '<GO>': 2, '<END>': 3, ' ': 4}

    data_sentences = []
    max_sent_len, max_word_len = -1, -1

    for sentence in list_of_list_of_words:
        words = []
        for word in sentence:
            letters = []
            for letter in word:
                if not is_ext_vocab and letter not in vocab:
                    vocab[letter] = len(vocab)
                if letter not in vocab:
                    token_id = vocab['<OOV>']
                else:
                    token_id = vocab[letter]
                letters.append(token_id)
            if len(letters) > max_word_len:
                max_word_len = len(letters)
            words.append(letters)
        if len(words) > max_sent_len:
            max_sent_len = len(words)
        data_sentences.append(words)

    print(data_sentences)
    print(np.shape(data_sentences))
    if max_word_len_ is not None:
        max_word_len = max_word_len_
    if max_sent_len_ is not None:
        max_sent_len = max_sent_len_

    enc_sentences = np.full([len(data_sentences), max_sent_len, max_word_len], vocab['<PAD>'], dtype=np.int32)
    dec_go_sentences = np.full([len(data_sentences), max_sent_len, max_word_len + 2], vocab['<PAD>'], dtype=np.int32)
    dec_end_sentences = np.full([len(data_sentences), max_sent_len, max_word_len + 2], vocab['<PAD>'], dtype=np.int32)

    # Create inputs for encoder, which will all be as long as the max_sent_length in the corpus, and padded
    # Also create inputs and labels for decoder. These will be 1 token longer to allow for GO and END tokens.
    sentence_lengths, word_lengths = [], []
    for i, sentence in enumerate(data_sentences):
        sentence_lengths.append(len(sentence))
        for j, word in enumerate(sentence):
            word_lengths.append(len(word))
            enc_sentences[i, j, 0:len(sentence)] = word
            dec_go_sentences[i, j, 0] = vocab['<GO>']
            dec_go_sentences[i, j, 1:len(sentence)+1] = word
            dec_go_sentences[i, j, len(sentence)+1] = vocab['<END>']
            dec_end_sentences[i, j, 0:len(sentence)] = sentence
            dec_end_sentences[i, len(sentence)] = vocab['<END>']

    sentence_lengths = np.array(sentence_lengths, dtype=np.int32)
    reverse_dictionary = dict(zip(vocab.values(), vocab.keys()))

    return vocab, reverse_dictionary, sentence_lengths, max_sent_len+1, enc_sentences, dec_go_sentences, dec_end_sentences







