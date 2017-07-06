import tensorflow as tf
import numpy as np
from util import import_data
from util import build_dictionary
from util import word_to_int
from skipthought import skipthought_model
import pandas as pd
import pickle as pkl
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse

from nltk.tokenize import word_tokenize
import os.path
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score as f1

def load_data(loc='./msrp/'):
    """
    Load MSRP dataset
    """

    trainA, trainB, testA, testB = [],[],[],[]
    trainS, devS, testS = [],[],[]

    trainloc = os.path.join(loc, 'msr_paraphrase_train.txt')
    f = open(trainloc, encoding="latin-1")
    for line in f:
        text = line.strip().split('\t')
        trainA.append(' '.join(word_tokenize(text[3])))
        trainB.append(' '.join(word_tokenize(text[4])))
        trainS.append(text[0])
    f.close()
    testloc = os.path.join(loc, 'msr_paraphrase_test.txt')
    f = open(testloc, encoding="latin-1")
    for line in f:
        text = line.strip().split('\t')
        testA.append(' '.join(word_tokenize(text[3])))
        testB.append(' '.join(word_tokenize(text[4])))
        testS.append(text[0])
    f.close()

    trainS = [int(s) for s in trainS[1:]]
    testS = [int(s) for s in testS[1:]]

    return [trainA[1:], trainB[1:]], [testA[1:], testB[1:]], [trainS, testS]

def feats(A, B):
    """
    Compute additional features (similar to Socher et al.)
    These alone should give the same result from their paper (~73.2 Acc)
    """
    tA = [t.split() for t in A]
    tB = [t.split() for t in B]
    
    nA = [[w for w in t if is_number(w)] for t in tA]
    nB = [[w for w in t if is_number(w)] for t in tB]

    features = np.zeros((len(A), 6))

    # n1
    for i in range(len(A)):
        if set(nA[i]) == set(nB[i]):
            features[i,0] = 1.

    # n2
    for i in range(len(A)):
        if set(nA[i]) == set(nB[i]) and len(nA[i]) > 0:
            features[i,1] = 1.

    # n3
    for i in range(len(A)):
        if set(nA[i]) <= set(nB[i]) or set(nB[i]) <= set(nA[i]): 
            features[i,2] = 1.

    # n4
    for i in range(len(A)):
        features[i,3] = 1.0 * len(set(tA[i]) & set(tB[i])) / len(set(tA[i]))

    # n5
    for i in range(len(A)):
        features[i,4] = 1.0 * len(set(tA[i]) & set(tB[i])) / len(set(tB[i]))

    # n6
    for i in range(len(A)):
        features[i,5] = 0.5 * ((1.0*len(tA[i]) / len(tB[i])) + (1.0*len(tB[i]) / len(tA[i])))

    return features

if __name__ == '__main__':
    _hidden_size = 64
    _learning_rate = 0.001
    _epochs = 500
    _batch_size = 64
    _train_size = 0.8
    _temp_size = 500

    traintext, testtext, labels = load_data()

    sent_all = traintext[0] + traintext[1]
    with open('./model/dict_files.pkl', 'rb') as f:
        dictionary, reverse_dictionary, dictionary_sorted = pkl.load(f)
    sent_lengths, _, sentences = word_to_int(sent_all[:_temp_size], dictionary)
    tf.reset_default_graph()
    model = skipthought_model(corpus = './corpus/gingerbread.txt',
        embedding_size = 200, 
        hidden_size = 200, 
        hidden_layers = 2, 
        batch_size = 32, 
        keep_prob_dropout = 1.0, 
        learning_rate = 0.005, 
        bidirectional = False,
        loss_function = 'softmax',
        sampled_words = 500,
        num_epochs = 100)
    model.load_model('./model/')
    sentences_encoded = model.encode(sentences, sent_lengths)
    n = np.shape(sentences_encoded)[1]//2
    
    sent_a = sentences_encoded[0, :n, :]
    sent_b = sentences_encoded[0, n:, :]
    feature_1 = sent_a * sent_b 
    feature_2 = np.abs(sent_a - sent_b)

    perm = np.random.permutation(n)
    n_train = int(np.floor(0.8*n))
    train_feature_1 = feature_1[perm][:n_train]
    train_feature_2 = feature_2[perm][:n_train]
    train_labels = np.expand_dims(labels[0],1).astype(float)[perm][:n_train]
    dev_feature_1 = feature_1[perm][n_train:]
    dev_feature_2 = feature_2[perm][n_train:]
    dev_labels = np.expand_dims(labels[0],1).astype(float)[perm][n_train:]

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        initializer = tf.random_normal_initializer()
        msrp_labels = tf.placeholder(tf.float32, [None,1], 'msrp_labels')
        msrp_feature_1 = tf.placeholder(tf.float32, [None, None], 'msrp_feature_1')
        msrp_feature_2 = tf.placeholder(tf.float32, [None, None], 'msrp_feature_2')  
        b_1 = tf.get_variable('msrp_bias_1', [_hidden_size], tf.float32, initializer = initializer)
        b_2 = tf.get_variable('msrp_bias_2', [1], tf.float32, initializer = initializer)
        W_1 = tf.get_variable('msrp_weight_1', [model.embedding_size, _hidden_size], tf.float32, initializer = initializer)
        W_2 = tf.get_variable('msrp_weight_2', [model.embedding_size, _hidden_size], tf.float32, initializer = initializer)
        W_3 = tf.get_variable('msrp_weight_3', [_hidden_size, 1], tf.float32, initializer = initializer)
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        hidden = tf.sigmoid(tf.matmul(msrp_feature_1, W_1) + tf.matmul(msrp_feature_2, W_2) + b_1)
        logits = tf.matmul(hidden, W_3) + b_2
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = msrp_labels, logits=logits))
        opt_op = tf.contrib.layers.optimize_loss(loss = loss, learning_rate = _learning_rate, 
            optimizer = 'Adam', global_step = global_step) 
        prediction = tf.nn.sigmoid(logits)

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run(session = sess)
            for epoch in range(_epochs):
                perm = np.random.permutation(n_train)
                feature_1_perm = train_feature_1[perm]
                feature_2_perm = train_feature_2[perm]
                labels_perm = train_labels[perm]
                avg_loss = 0
                for step in range(n//_batch_size):
                    begin = step * _batch_size
                    end = (step + 1) * _batch_size
                    batch_feature_1 = feature_1_perm[begin : end]
                    batch_feature_2 = feature_2_perm[begin : end]
                    batch_labels = labels_perm[begin : end]
                    train_dict = {msrp_labels: batch_labels,
                                  msrp_feature_1: batch_feature_1,
                                  msrp_feature_2: batch_feature_2}
                    _, batch_loss, batch_prediction = sess.run([opt_op, loss, prediction], feed_dict=train_dict)
                    avg_loss += batch_loss
                if epoch % 10==0:
                    dev_dict = {msrp_labels: dev_labels,
                            msrp_feature_1: dev_feature_1,
                            msrp_feature_2: dev_feature_2}
                    _, dev_loss, dev_prediction = sess.run([opt_op, loss, prediction], feed_dict=dev_dict)
                    pr = pearsonr(dev_prediction[:,0], dev_labels[:,0])[0]
                    sr = spearmanr(dev_prediction[:,0], dev_labels[:,0])[0]
                    se = mse(dev_prediction[:,0], dev_labels[:,0])
                    print('Epoch %d: Train loss: %0.2f, Dev loss: %0.2f, Dev pearson: %0.2f, Dev spearman: %0.2f, Dev MSE: %0.2f\n' % (epoch, avg_loss, dev_loss, pr, sr, se))


