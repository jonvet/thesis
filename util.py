import nltk
import tensorflow as tf
import numpy as np


def import_data(path):

    '''
    Imports data and tokenises data.
    Outputs a list of sentences, where each sentence is itself a list of words
    '''

    sentences = []
    words = []
    with open(path, 'r') as myfile:
        data = myfile.read().replace('\n', ' ')
    sentences = nltk.sent_tokenize(data)
    for sentence in sentences:
        this_sentence = nltk.word_tokenize(sentence)
        words.append(this_sentence)
    return words

def word_to_int(sentences, vocab, max_sent_len_=None):

    '''
    Encodes sentences using the vocabulary provided
    '''

    data_sentences = []
    max_sent_len = -1
    for sentence in sentences:
        words = []
        for word in sentence:
            if word not in vocab:
                token_id = vocab['<OOV>']
            else:
                token_id = vocab[word]
            words.append(token_id)
        if len(words) > max_sent_len:
            max_sent_len = len(words)
        data_sentences.append(words)
    if max_sent_len_ is not None:
        max_sent_len = max_sent_len_
    enc_sentences = np.full([len(data_sentences), max_sent_len], vocab['<PAD>'], dtype=np.int32)
    sentence_lengths = []
    for i, sentence in enumerate(data_sentences):
        sentence_lengths.append(len(sentence))
        enc_sentences[i, 0:len(sentence)] = sentence
    sentence_lengths = np.array(sentence_lengths, dtype=np.int32)

    return sentence_lengths, max_sent_len, enc_sentences

def build_dictionary(sentences, vocab=None, max_sent_len_=None):

    is_ext_vocab = True

    # If no vocab provided, create a new one
    if vocab is None:
        is_ext_vocab = False
        vocab = {'<PAD>': 0, '<OOV>': 1, '<GO>': 2, '<END>': 3}

    # Create list that will contain integer encoded senteces 
    data_sentences = []
    max_sent_len = -1

    for sentence in sentences:
        words = []
        for word in sentence:
            # If creating a new vocab, and word isnt in vocab yet, add it
            if not is_ext_vocab and word not in vocab:
                vocab[word] = len(vocab)
            # Now add either OOV or actual token to words
            if word not in vocab:
                token_id = vocab['<OOV>']
            else:
                token_id = vocab[word]
            words.append(token_id)
        if len(words) > max_sent_len:
            max_sent_len = len(words)
        data_sentences.append(words)

    if max_sent_len_ is not None:
        max_sent_len = max_sent_len_

    enc_sentences = np.full([len(data_sentences), max_sent_len], vocab['<PAD>'], dtype=np.int32)
    dec_go_sentences = np.full([len(data_sentences), max_sent_len + 2], vocab['<PAD>'], dtype=np.int32)
    dec_end_sentences = np.full([len(data_sentences), max_sent_len + 2], vocab['<PAD>'], dtype=np.int32)

    # Create inputs for encoder, which will all be as long as the max_sent_length in the corpus, and padded
    # Also create inputs and labels for decoder. These will be 1 token longer to allow for GO and END tokens.
    sentence_lengths = []
    for i, sentence in enumerate(data_sentences):
        sentence_lengths.append(len(sentence))
        enc_sentences[i, 0:len(sentence)] = sentence
        dec_go_sentences[i, 0] = vocab['<GO>']
        dec_go_sentences[i, 1:len(sentence)+1] = sentence
        dec_go_sentences[i, len(sentence)+1] = vocab['<END>']
        dec_end_sentences[i, 0:len(sentence)] = sentence
        dec_end_sentences[i, len(sentence)] = vocab['<END>']

    sentence_lengths = np.array(sentence_lengths, dtype=np.int32)
    reverse_dictionary = dict(zip(vocab.values(), vocab.keys()))

    # enc_sentences = enc_sentences[:,:5]
    # dec_go_sentences = dec_go_sentences[:,:5]
    # dec_end_sentences = dec_end_sentences[:,:5]
    # max_sent_len = 4

    return vocab, reverse_dictionary, sentence_lengths, max_sent_len+1, enc_sentences, dec_go_sentences, dec_end_sentences

# words, sentences = import_data('finn.txt')
# vocab, reverse_dictionary, sentence_lengths, max_sent_len, enc_sentences, dec_go_sentences, dec_end_sentences = build_dictionary(words)