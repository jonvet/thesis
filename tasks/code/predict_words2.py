import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import os
import random
from collections import defaultdict

class Predict_words2(object):

    def __init__(self, 
        vocab, 
        sent_dim = 4096,
        word_dim = 300,
        learning_rate = 0.001,
        decay_steps = 1000,
        decay = 1.0,
        batch_size = 64):

        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        self.sent_dim = sent_dim
        self.word_dim = word_dim
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay = decay
        self.batch_size = batch_size
        self.labels_list = ['Negative', 'Positive']
        self.labels_dict = {'Negative':0, 'Positive': 1}

        self.graph = tf.get_default_graph()
        self.initializer = tf.random_normal_initializer()

        self.X = tf.placeholder(
            tf.float32, 
            [None, self.sent_dim], 
            'sentences')
        self.words = tf.placeholder(
            tf.float32, 
            [None, self.word_dim], 
            'words')
        self.y = tf.placeholder(
            tf.float32, 
            [None, 1], 
            'labels')
        self.global_step = tf.Variable(0, 
            name = 'global_step', 
            trainable = False)

        self.features = tf.concat([self.X, self.words], axis = 1)

        self.logits = tf.contrib.layers.fully_connected(
            self.features, 1, activation_fn=None)

        self.prediction = tf.cast(tf.round(tf.sigmoid(self.logits)), tf.int32)

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                labels = self.y, 
                logits = self.logits))

        # self.loss = tf.reduce_mean(tf.square(self.l))

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
    
    def get_accuracy(self, data, labels, prediction):

        correct = np.sum(labels==prediction)
        accuracy = correct/len(labels)
        return accuracy

    def get_f1_pre_rec(self, labels, prediction):

        pre, rec, f1, _ = precision_recall_fscore_support(
            y_true = labels, 
            y_pred = prediction,
            labels = [self.labels_dict[i] for i in self.labels_list])

        counts = np.zeros([2, 1])
        for i in labels:
            counts[i] += 1

        return np.expand_dims(pre,1), np.expand_dims(rec,1), np.expand_dims(f1,1), counts

    def get_confusion(self, labels, prediction):

        matrix = confusion_matrix(labels, prediction,
            labels = [self.labels_dict[i] for i in self.labels_list])
        return matrix

    def get_pos_samples(self, data):

        pos_samples= [[random.sample(sentence.split(' ')[:-1], 1)[0] for sentence in part] for part in data]
        return pos_samples

    def get_neg_samples(self, data, pos_labels):

        unique_pos = set(pos_labels[0]+pos_labels[1]+pos_labels[2])
        neg_samples = []
        for j, part in enumerate(data):
            part_list = []
            for i, sentence in enumerate(part):
                tokenized_sentence = sentence.split(' ')[:-1]
                done = False
                while not done:
                    candidate = random.sample(unique_pos, 1)[0]
                    done = candidate != pos_labels[j][i]
                # candidates = [word for word in unique_pos if word != pos_labels[j][i]]
                # part_list.append(random.sample(candidates, 1))

                part_list.append(candidate)
            neg_samples.append(part_list)    
        return neg_samples

    def get_embedding(self, word):

        if self.embeddings == None:
            return self.vocab[word]
        else:
            return self.embeddings[self.vocab[word],:]

    def run_batch(self, data, string_sentences, samples, train = False, mclasses = None):
        

        sentences = []
        words = []
        labels = []
        # print(len(samples[0]), len(samples[1]), len(data[0]))
        for i in range(len(samples[0])):
            try:
                pos = self.vocab[samples[0][i]]
                neg = self.vocab[samples[1][i]]
                words.append(pos)
                words.append(neg)
                sentences.append(data[0][i])
                sentences.append(data[0][i])
                labels.append(1)
                labels.append(0)
            except:
                next
        # print(len(words), len(sentences), len(labels))
        # sentences = data[0] + data[0]
        # pos_embeddings = [self.vocab[word] for word in labels[0]]
        # neg_embeddings = [self.vocab[word] for word in labels[1]]
        # words = pos_embeddings + neg_embeddings
        # labels = [1] * len(pos_embeddings) + [0]*len(neg_embeddings)

        sentences = np.array(sentences)
        words = np.array(words)
        labels = np.expand_dims(labels,1)

        feed_dict = {self.X: sentences,
                    self.words: words,
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

