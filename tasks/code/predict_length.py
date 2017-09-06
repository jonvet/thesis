import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import os

class Predict_length(object):

    def __init__(self,
        dim = 4096,
        learning_rate = 0.0001,
        decay_steps = 1000,
        decay = 1.0,
        batch_size = 64):

        self.dim = dim
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay = decay
        self.batch_size = batch_size
        self.labels_list = ['0-4', '5-6' ,'7-8' ,'9-10' ,'11-12', '14-60']
        self.labels_dict = {}
        for i,rel in enumerate(self.labels_list):
            self.labels_dict[rel] = i

        self.graph = tf.get_default_graph()
        self.initializer = tf.random_normal_initializer()

        self.X = tf.placeholder(
            tf.float32, 
            [None, self.dim], 
            'sentences')
        self.y = tf.placeholder(
            tf.int32, 
            [None], 
            'labels')
        self.global_step = tf.Variable(0, 
            name = 'global_step', 
            trainable = False)

        self.logits = tf.contrib.layers.fully_connected(
            self.X, 7, activation_fn=None)
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
        saver = tf.train.Saver()
        saver.save(sess = self.sess, save_path = path + '/step_%d' % step, write_state = False)

    def load_model(self, path, step):

        saver = tf.train.Saver()
        saver.restore(self.sess, path + '/step_%d' % step)
        print('Sentence length model restored')

