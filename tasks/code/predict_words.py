import tensorflow as tf
import numpy as np
from collections import defaultdict

class Predict_words(object):

    def __init__(self, 
        vocab, 
        epochs = 100,
        dim = 4096,
        learning_rate = 0.001,
        decay_steps = 1000,
        decay = 1.0,
        batch_size = 64):

        self.vocab = vocab
        self.vocab_size = len(self.vocab)
        self.ordered_dict = defaultdict(int)
        for w in self.vocab.keys():
            self.ordered_dict[w] = len(self.ordered_dict)
        self.epochs = epochs
        self.dim = dim
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay = decay
        self.batch_size = batch_size

        self.graph = tf.get_default_graph()
        self.initializer = tf.random_normal_initializer()

        self.X = tf.placeholder(
            tf.float32, 
            [None, self.dim], 
            'sentences')
        self.y = tf.placeholder(
            tf.float32, 
            [None, self.vocab_size], 
            'labels')
        self.global_step = tf.Variable(0, 
            name = 'global_step', 
            trainable = False)

        self.logits = tf.contrib.layers.fully_connected(
            self.X, self.vocab_size, activation_fn=None)
        self.prediction = tf.sigmoid(self.logits)

        self.l = tf.nn.sigmoid_cross_entropy_with_logits(
                labels = self.y, 
                logits = self.logits)

        self.loss = tf.reduce_mean(tf.square(self.l))

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
            global_step = self.global_step) 
    
    def get_labels(self, data, sentences):
        sentences, _ = data
        labels = np.zeros([len(sentences), len(self.vocab)])
        for i, sentence in enumerate(sentences):
            for word in sentence:
                labels[i, self.ordered_dict[word]] = 1
        return np.squeeze(labels)

    def get_accuracy(self, data, labels, prediction):

        lengths = data[1]
        preds = [np.argpartition(p, -l)[-l:] for l,p in zip(lengths, prediction)]
        truths = [np.where(d==1) for d in labels]
        total = 0
        correct = 0
        for pred, truth in zip(preds, truths):
                for w in pred:
                    total += 1
                    if w in truth[0]:
                        correct += 1
        accuracy = correct/total
        return accuracy

    def run_batch(self, data, sentences, labels, train = False):
        
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
            return batch_prediction, batch_loss