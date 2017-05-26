import tensorflow as tf
import numpy as np
from util import import_data
from util import build_dictionary
from util import word_to_int
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.tensorboard.plugins import projector
import time
from sys import stdout
import os
import operator
import csv
from skipthought import skipthought
import pandas as pd
import pickle as pkl
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse

def encode_labels(labels, nclass=5):
    """
    Label encoding from Tree LSTM paper (Tai, Socher, Manning)
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y    

if __name__ == '__main__':
    _hidden_size = 64
    _learning_rate = 0.001
    _epochs = 500
    _batch_size = 64
    _train_size = 0.8
    _temp_size = 1000

    data = pd.read_csv('./sick/sick_train/SICK_train.txt', sep='\t', index_col=0)
    sent_all = data.sentence_A.tolist() + data.sentence_B.tolist()
    with open('./model/dict_files.pkl', 'rb') as f:
        dictionary, reverse_dictionary, dictionary_sorted = pkl.load(f)
    sent_lengths, _, sentences = word_to_int(sent_all[:_temp_size], dictionary)
    tf.reset_default_graph()
    model = skipthought(corpus = './corpus/gingerbread.txt',
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

    score = data.relatedness_score.tolist()
    score_encoded = encode_labels(score[:_temp_size])

    perm = np.random.permutation(n)
    n_train = int(np.floor(0.8*n))
    train_feature_1 = feature_1[perm][:n_train]
    train_feature_2 = feature_2[perm][:n_train]
    train_score_encoded = score_encoded[perm][:n_train]
    dev_feature_1 = feature_1[perm][n_train:]
    dev_feature_2 = feature_2[perm][n_train:]
    dev_score_encoded = score_encoded[perm][n_train:]
    dev_score = np.array(score)[perm][n_train:]

    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        initializer = tf.random_normal_initializer()
        sick_scores = tf.placeholder(tf.int32, [None, None], 'sick_scores')
        sick_feature_1 = tf.placeholder(tf.float32, [None, None], 'sick_feature_1')
        sick_feature_2 = tf.placeholder(tf.float32, [None, None], 'sick_feature_2')  
        b_1 = tf.get_variable('sick_bias_1', [_hidden_size], tf.float32, initializer = initializer)
        b_2 = tf.get_variable('sick_bias_2', [5], tf.float32, initializer = initializer)
        W_1 = tf.get_variable('sick_weight_1', [model.embedding_size, _hidden_size], tf.float32, initializer = initializer)
        W_2 = tf.get_variable('sick_weight_2', [model.embedding_size, _hidden_size], tf.float32, initializer = initializer)
        W_3 = tf.get_variable('sick_weight_3', [_hidden_size, 5], tf.float32, initializer = initializer)
        r = tf.reshape(tf.range(1, 6, 1, dtype = tf.float32), [-1, 1])
        global_step = tf.Variable(0, name = 'global_step', trainable = False)
        hidden = tf.sigmoid(tf.matmul(sick_feature_1, W_1) + tf.matmul(sick_feature_2, W_2) + b_1)
        logits = tf.matmul(hidden, W_3) + b_2
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = sick_scores, logits=logits))
        opt_op = tf.contrib.layers.optimize_loss(loss = loss, learning_rate = _learning_rate, 
            optimizer = 'Adam', global_step = global_step) 
        prediction = tf.matmul(tf.nn.softmax(logits), r)

        with tf.Session(graph=graph) as sess:
            tf.global_variables_initializer().run(session = sess)
            for epoch in range(_epochs):
                perm = np.random.permutation(n_train)
                feature_1_perm = train_feature_1[perm]
                feature_2_perm = train_feature_2[perm]
                score_encoded_perm = train_score_encoded[perm]
                avg_loss = 0
                for step in range(n//_batch_size):
                    begin = step * _batch_size
                    end = (step + 1) * _batch_size
                    batch_feature_1 = feature_1_perm[begin : end]
                    batch_feature_2 = feature_2_perm[begin : end]
                    batch_score_encoded = score_encoded_perm[begin : end]
                    train_dict = {sick_scores: batch_score_encoded,
                                  sick_feature_1: batch_feature_1,
                                  sick_feature_2: batch_feature_2}
                    _, batch_loss, batch_prediction = sess.run([opt_op, loss, prediction], feed_dict=train_dict)
                    avg_loss += batch_loss
                if epoch % 10==0:
                    dev_dict = {sick_scores: dev_score_encoded,
                            sick_feature_1: dev_feature_1,
                            sick_feature_2: dev_feature_2}
                    _, dev_loss, dev_prediction,l,h = sess.run([opt_op, loss, prediction,logits, hidden], feed_dict=dev_dict)
                    pr = pearsonr(dev_prediction[:,0], dev_score)[0]
                    sr = spearmanr(dev_prediction[:,0], dev_score)[0]
                    se = mse(dev_prediction[:,0], dev_score)
                    print('Epoch %d: Train loss: %0.2f, Dev loss: %0.2f, Dev pearson: %0.2f, Dev spearman: %0.2f, Dev MSE: %0.2f\n' % (epoch, avg_loss, dev_loss, pr, sr, se))

