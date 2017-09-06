import collections
import os.path
import gensim.models
import numpy as np
import sklearn.linear_model
import tensorflow as tf
import codecs
from gensim.models import word2vec
import pickle as pkl
from skipthought import Skipthought_para
from skipthought import Skipthought_model
import operator

# def get_glove_vocab(filename):
#     vocab = collections.OrderedDict()
#     numpy_arrays = []
#     labels_array = []
#     with codecs.open(vector_file, 'r', 'utf-8') as f:
#         for c, r in enumerate(f):
#             try:
#                 sr = r.split()
#                 labels_array.append(sr[0])
#                 numpy_arrays.append( np.array([float(i) for i in sr[1:]]) )
#             except:
#                 print()
#                 next
#             if c == n_words:
#                 return np.array( numpy_arrays ), labels_array
#     return np.array( numpy_arrays ), labels_array

def load_skip_thoughts_embeddings(path, step):

    with open(path + 'vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    tf.reset_default_graph()
    model = Skipthought_model(
        vocab = vocab, 
        parameters = paras, 
        path = path)
    model.load_model(path, step)
    embeddings = model.word_embeddings.eval(session = model.sess)

    return embeddings

def _expand_vocabulary(skip_thoughts_emb, skip_thoughts_vocab, word2vec):

    # Find words shared between the two vocabularies.
    print("Finding shared words")
    shared_words = [w for w in word2vec.vocab if w in skip_thoughts_vocab]

    # Select embedding vectors for shared words.
    print("Selecting embeddings for %d shared words" % len(shared_words))
    shared_st_emb = skip_thoughts_emb[[
        skip_thoughts_vocab[w] for w in shared_words]]
    shared_w2v_emb = word2vec[shared_words]

    # Train a linear regression model on the shared embedding vectors.
    print("Training linear regression model")
    model = sklearn.linear_model.LinearRegression()
    model.fit(shared_w2v_emb, shared_st_emb)

    # Create the expanded vocabulary.
    print("Creating embeddings for expanded vocabulary")
    embedding_map = collections.OrderedDict()
    print('Length of word2vec vocabulary: %d\n' % len(word2vec.vocab))
    for i, w in enumerate(word2vec.vocab):
        print('\rEmbedding %d' %(i+1), end='   ')
    # Ignore words with underscores (spaces).
        if "_" not in w:
            w_emb = model.predict(word2vec[w].reshape(1, -1))
            embedding_map[w] = w_emb.reshape(-1)

    for w in skip_thoughts_vocab:
        embedding_map[w] = skip_thoughts_emb[skip_thoughts_vocab[w]]

    print("Created expanded vocabulary of %d words", len(embedding_map))

    expanded_vocab = {}
    expanded_embeddings = np.zeros([len(embedding_map), paras.embedding_size])

    for i, w in enumerate(embedding_map.keys()):
        expanded_vocab[w] = i
        expanded_embeddings[i,:] = embedding_map[w]

    print('Saving expanded vocab and embeddings')
    with open(path + 'expanded_vocab.pkl', 'wb') as f:
        pkl.dump(expanded_vocab, f)

    embeddings_file = os.path.join(path, "expanded_embeddings.npy")
    np.save(embeddings_file, expanded_embeddings)

    return expanded_vocab, expanded_embeddings

# path = '../models/toronto_n5/'
path = '/cluster/project2/mr/vetterle/skipthought/toronto_n13/'

print('Loading trained skipthought word embeddings')
with open(path + 'paras.pkl', 'rb') as f:
    paras = pkl.load(f)
skipthought_embeddings = load_skip_thoughts_embeddings(
    path = path, 
    step = 425000)
with open(path + 'vocab.pkl', 'rb') as f:
    skipthought_vocab = pkl.load(f)

# if not (os.path.exists(path+'expanded_vocab.pkl') & os.path.exists(path+'expanded_embeddings.npy')):
    
print('Loading skipthought vocabulary')
sorted_vocab = sorted(skipthought_vocab.items(),
    key=operator.itemgetter(1))
skipthought_vocab = collections.OrderedDict(sorted_vocab)

print('Loading word2vec word embeddings')
# word2vec = gensim.models.KeyedVectors.load_word2vec_format('/Users/Jonas/Documents/dev/GoogleNews-vectors-negative300.bin', binary=True) 
word2vec = gensim.models.KeyedVectors.load_word2vec_format('/cluster/project6/mr_corpora/vetterle/word2vec/GoogleNews-vectors-negative300.bin', binary=True)

expanded_vocab, expanded_embeddings = _expand_vocabulary(
    skipthought_embeddings, 
    skipthought_vocab,
    word2vec)
# else: 
#     print('Expanded vocab and embeddings already exist in %s' % path)

# encas = model.encoded_sentences.eval(session = model.sess, feed_dict={model.graph.get_tensor_by_name('embedding_lookup:0'): test, model.sentences_lengths: test_lengths})
