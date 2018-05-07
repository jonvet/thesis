import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import os
from infersent.data import get_nli
import pickle as pkl
import random
from collections import defaultdict
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

class Predict_length(object):

    def __init__(self,
        encoder,
        learning_rate = 0.0001,
        decay_steps = 1000,
        decay = 1.0,
        batch_size = 64,
        epochs = 10):

        self.encoder = encoder
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay = decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.sent_dim = encoder.sent_dim

        self.labels_list = ['0-4', '5-6' ,'7-8' ,'9-10' ,'11-12', '14-60']
        self.labels_dict = {}
        for i,rel in enumerate(self.labels_list):
            self.labels_dict[rel] = i

        self.graph = tf.get_default_graph()
        self.initializer = tf.random_normal_initializer()

        self.X = tf.placeholder(
            tf.float32, 
            [None, self.sent_dim], 
            'sentences')
        self.y = tf.placeholder(
            tf.int32, 
            [None], 
            'labels')
        self.global_step = tf.Variable(0, 
            name = 'global_step', 
            trainable = False)

        self.logits = tf.contrib.layers.fully_connected(
            self.X, 6, activation_fn=None, scope='output_layer')
        self.prediction = tf.argmax(self.logits, 1)

        self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = self.y, 
                logits = self.logits)

        self.loss = tf.reduce_mean(self.loss)

        self.eta = tf.train.exponential_decay(
            self.learning_rate, 
            self.global_step, 
            self.decay_steps, 
            self.decay, 
            staircase=True)
        self.opt_op = tf.contrib.layers.optimize_loss(
            loss = self.loss, 
            learning_rate = self.eta, 
            optimizer = 'Adam', 
            global_step = self.global_step,
            clip_gradients = 5.0) 
    
    def get_labels(self, data, sentences):

        '''
        Goldberg et al use bins
        (5-8), (9-12), (13-16), (17-20), (21-25), (26-29), (30-33), (34-70)
        '''
        _, lengths = data 
        bins = np.array([0, 5, 7,9 , 11, 13, 70])
        labels = np.digitize(lengths, bins) - np.ones_like(lengths)
        # print(labels)
        # print(self.labels_dict)
        # print(self.labels_list)
        return labels

    def get_accuracy(self, data, labels, prediction):

        correct = np.sum(labels==prediction)
        accuracy = correct/len(labels)
        return accuracy

    def get_f1_pre_rec(self, labels, prediction):

        pre, rec, f1, _ = precision_recall_fscore_support(
            y_true = labels, 
            y_pred = prediction,
            labels = [self.labels_dict[i] for i in self.labels_list])

        counts = np.zeros([6, 1])
        for i in labels:
            counts[i] += 1

        return np.expand_dims(pre,1), np.expand_dims(rec,1), np.expand_dims(f1,1), counts

    def get_confusion(self, labels, prediction):

        matrix = confusion_matrix(labels, prediction,
            labels = [self.labels_dict[i] for i in self.labels_list])
        return matrix

    def run_batch(self, data, sentences, labels, train = False, mclasses = None):
        
        sentences, _ = data
        feed_dict = {self.X: sentences,
                      self.y: labels}

        if train:
            _, batch_loss = self.sess.run(
                [self.opt_op, self.loss], 
                feed_dict=feed_dict)
            return batch_loss

        else:
            batch_prediction, batch_loss = self.sess.run(
                [self.prediction, self.loss], 
                feed_dict=feed_dict)
            return batch_prediction, batch_loss, labels

    def save_model(self, path, step):

        if not os.path.exists(path):
            os.makedirs(path)
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer'))
        saver.save(sess = self.sess, save_path = path + '/step_%d' % step)

    def load_model(self, path, step):

        saver = tf.train.Saver()
        saver.restore(self.sess, path + '/step_%d' % step)
        print('Sentence length model restored')

    def train_model(self, X_train, X_dev, y_train, y_dev, save_path):

        self.sess = tf.Session(graph = self.graph)
        tf.global_variables_initializer().run(session = self.sess)

        best_f1 = 0
        for epoch in range(self.epochs):
            print('\nStarting epoch %d' % epoch)
            perm = np.random.permutation(len(X_train))
            train_perm = X_train[perm]

            y_dev = None
            avg_loss = 0
            steps = len(X_train) // self.batch_size

            for step in range(0, len(X_train), self.batch_size):

                sentences = train_perm[step:(step+self.batch_size)]
                sentence_data = self.encoder.embed(sentences)
                labels = self.get_labels(sentence_data, sentences)
                loss = self.run_batch(sentence_data, sentences, labels, train=True)
                avg_loss += loss/steps
                print('\rBatch loss at step %d: %0.5f' % (step/self.batch_size, loss), end = '    ')
               
            _,_,f1 = self.test_model(X_dev, y_dev)

            if f1>best_f1:
                self.save_model(save_path, 1)   
            else:
                break

    def test_model(self, X, y, step = None, saved_model_path = None, mclasses = None):

        if saved_model_path != None:
            self.sess = tf.Session(graph = self.graph)
            self.load_model(saved_model_path, step)

        num_classes = len(self.labels_list)
        dev_loss, dev_accuracy = 0, 0
        dev_f1, dev_pre, dev_rec, dev_counts = np.zeros([num_classes, 1]), np.zeros([num_classes, 1]), np.zeros([num_classes, 1]), np.zeros([num_classes, 1])
        dev_confusion = np.zeros([num_classes, num_classes])
        dev_steps = len(X) // self.batch_size
        all_labels, all_preds = [], []

        for step in range(0, len(X), self.batch_size):

            print('\rStep {}/{}'.format(step/self.batch_size, dev_steps), end = '    ')
            sentences = X[step:(step+self.batch_size)]
            sentence_data = self.encoder.embed(sentences)
            labels = self.get_labels(sentence_data, sentences)
            prediction, loss, labels = self.run_batch(sentence_data, sentences, labels, train=False, mclasses=mclasses)
            accuracy = self.get_accuracy(sentence_data, labels, prediction)
            dev_accuracy += accuracy/dev_steps
            all_labels += list(labels)
            all_preds += list(prediction)

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        dev_pre, dev_rec, dev_f1, dev_counts = self.get_f1_pre_rec(all_labels, all_preds)
        matrix = self.get_confusion(all_labels, all_preds)

        print('Test accuracy: %0.4f\n' % dev_accuracy)
        weights = dev_counts/np.sum(dev_counts)
        weights_no_none = dev_counts[1:]/np.sum(dev_counts[1:])
        weighted_f1 = float(np.matmul(weights.T,dev_f1))
        weighted_f1_no_none = float(np.matmul(weights_no_none.T,dev_f1[1:]))
        print('Weighted F1 score: %0.4f\n' % weighted_f1)
        print('Weighted F1 score (no none): %0.4f\n' % weighted_f1_no_none)

        df_accuracy = pd.DataFrame(data=np.concatenate((dev_f1, dev_pre, dev_rec, dev_counts), axis=1),
            columns=['F1', 'Precision', 'Recall', 'n'], index=self.labels_list)
        print(df_accuracy)
        # return dev_confusion, df_accuracy, weighted_f1
        return dev_confusion, df_accuracy, weighted_f1_no_none

def setup(snli_path, toy=False):

    print('Loading corpus')
    train, dev, test = get_nli(snli_path)
    train = np.array(train['s2'])
    dev = np.array(dev['s2'])
    test = np.array(test['s2'])

    if toy:
        train = train[:500]
        dev = dev[:500]
        test = test[:500]

    return train, dev, test

def balance_data(data):

    lengths = [len(s.split(' ')) for s in data]
    data = data[np.array(lengths)<=70]
    lengths = [len(s.split(' ')) for s in data]
    bins = np.array([0, 5, 8, 12, 17, 21, 26, 70])
    share_dev = 0.05
    labels = np.digitize(lengths, bins) - np.ones_like(lengths)
    sss = StratifiedShuffleSplit(n_splits=2, test_size=0.05, random_state=0)
    for train_index, test_index in sss.split(lengths, labels):
        X_train, X_test = data[train_index], data[test_index]
    return X_train, X_test

def length_summary(data):

    lengths = lengths = [len(s.split()) for s in data]