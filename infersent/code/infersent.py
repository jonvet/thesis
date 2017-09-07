import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
import time
from sys import stdout
import os
import glob
import operator
import csv
import pickle as pkl
from collections import defaultdict
from data import get_nli
from data import build_vocab
import shutil

class Infersent_para(object):

    def __init__(self, embedding_size, hidden_size, hidden_layers, batch_size, keep_prob_dropout, learning_rate, 
            bidirectional, decay, lrshrink, eval_step, uniform_init_scale, clip_gradient_norm, save_every, epochs):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.keep_prob_dropout = keep_prob_dropout
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.decay = decay
        self.lrshrink = lrshrink
        self.eval_step = eval_step
        self.uniform_init_scale = uniform_init_scale
        self.clip_gradient_norm = clip_gradient_norm
        self.save_every = save_every
        self.epochs = epochs

class Infersent_model(object):

    def __init__(self, vocab, parameters, path):
        self.para = parameters
        self.vocab = vocab
        self.vocabulary_size = len(self.vocab)
        self.path = path
        self.learning_rate = self.para.learning_rate

        print('\r~~~~~~~ Building graph ~~~~~~~\r')
        self.graph = tf.get_default_graph()
        self.initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        self.uniform_initializer = tf.random_uniform_initializer(
            minval=-self.para.uniform_init_scale, 
            maxval=self.para.uniform_init_scale)

        self.global_step = tf.Variable(0, 
            name = 'global_step', 
            trainable = False)
        self.eta = tf.placeholder(tf.float32, 
            [],
            name = "eta")
        self.s1_embedded = tf.placeholder(tf.float32, 
            [None, None, self.para.embedding_size], 
            "s1_embedded")
        self.s1_lengths = tf.placeholder(tf.int32, 
            [None], 
            "s1_embedded")
        self.s2_embedded = tf.placeholder(tf.float32, 
            [None, None, self.para.embedding_size], 
            "s2_embedded")
        self.s2_lengths = tf.placeholder(tf.int32, 
            [None], 
            "s2_embedded")
        self.labels = tf.placeholder(tf.int32, 
            [None], 
            "labels")

        with tf.variable_scope("encoder") as varscope:
            cell = tf.contrib.rnn.LSTMCell(self.para.hidden_size)
            cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = self.para.keep_prob_dropout)

            s1_sentences_states, s1_last_state = tf.nn.bidirectional_dynamic_rnn(
                cell, cell, 
                inputs = self.s1_embedded, 
                sequence_length = self.s1_lengths, 
                dtype=tf.float32, 
                scope = varscope)

            s1_states_fw, s1_states_bw = s1_sentences_states
            self.s1_states_h = tf.concat([s1_states_fw, s1_states_bw], axis = 2)
            self.s1_states_h = tf.reduce_max(self.s1_states_h, axis=1)

            varscope.reuse_variables()

            s2_sentences_states, s2_last_state = tf.nn.bidirectional_dynamic_rnn(
                cell, cell, 
                inputs = self.s2_embedded, 
                sequence_length = self.s2_lengths, 
                dtype=tf.float32, 
                scope = varscope)

            s2_states_fw, s2_states_bw = s2_sentences_states
            self.s2_states_h = tf.concat([s2_states_fw, s2_states_bw], axis = 2)
            self.s2_states_h = tf.reduce_max(self.s2_states_h, axis=1)

        with tf.variable_scope("classification_layer") as varscope:
            self.features = tf.concat(
                [self.s1_states_h, 
                self.s2_states_h, 
                tf.abs(self.s1_states_h - self.s2_states_h), 
                self.s1_states_h * self.s2_states_h],
                axis = 1)
            hidden = tf.contrib.layers.fully_connected(
                inputs = self.features,
                num_outputs= 512)
            logits = tf.contrib.layers.fully_connected(
                inputs = hidden,
                activation_fn = None,
                num_outputs= 3)

        with tf.variable_scope("loss") as varscope:
            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels = self.labels,
                logits = logits))
            self.opt_op = tf.contrib.layers.optimize_loss(
                loss = self.loss, 
                global_step = self.global_step, 
                learning_rate = self.eta, 
                optimizer = 'SGD', 
                clip_gradients=self.para.clip_gradient_norm, 
                learning_rate_decay_fn=None,
                summaries=None)

            self.loss_sum = tf.summary.scalar('loss', self.loss)
            self.lr_sum = tf.summary.scalar('learning_rate', self.eta)

        with tf.name_scope('accuracy'):
            pred = tf.argmax(logits,1)
            correct_prediction = tf.equal(self.labels, tf.cast(pred, tf.int32))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            self.acc_sum = tf.summary.scalar('accuracy', self.accuracy)

    def save_model(self, path, step):
        if not os.path.exists(path):
            os.mkdir(path)
        self.saver.save(sess = self.sess, save_path = path + '/step_%d' % step, write_state = False)
        print('Model saved')

    def load_model(self, path, step):
        self.sess = tf.Session(graph = self.graph)
        saver = tf.train.Saver()
        saver.restore(self.sess, path + '/saved_models/step_%d' % step)
        print('Model restored')

    def initialise(self):
        
        self.train_loss_writer = tf.summary.FileWriter(self.path +'tensorboard/train_loss', self.sess.graph)
        self.dev_loss_writer = tf.summary.FileWriter(self.path +'tensorboard/dev_loss', self.sess.graph)
        self.dev_accuracy_writer = tf.summary.FileWriter(self.path +'tensorboard/dev_accuracy', self.sess.graph)
        self.train_summary = tf.summary.merge([self.loss_sum, self.lr_sum])
        self.dev_loss_summary = tf.summary.merge([self.loss_sum])
        self.dev_accuracy_summary = tf.summary.merge([self.acc_sum])
        self.saver = tf.train.Saver()
        self.start_time = time.time()
        self.dev_accuracies = []
        self.dev_loss = []

    def get_batch(self, batch):

        '''
        Embeds the words of sentences using word2vec
        '''

        lengths = np.array([len(x) for x in batch])
        max_len = np.max(lengths)
        embed = np.zeros((len(batch), max_len, self.para.embedding_size))
        for i in range(len(batch)):
            for j in range(len(batch[i])):
                embed[i, j, :] = self.vocab[batch[i][j]]
        return embed, lengths
    
    def train(self, train_data, dev_data):
        try:
            print('\n~~~~~~~ Starting training ~~~~~~~\n')
            train_loss = 0
            self.dev_accuracy = -np.inf
            train_length = len(train_data['s1'])  
            dev_length = len(dev_data['s1'])  

            for epoch in range(self.para.epochs):
                print('\nEpoch %d, shuffling data...\n' % epoch)
                perm = np.random.permutation(train_length)
                train_s1 = np.array(train_data['s1'])[perm]
                train_s2 = np.array(train_data['s2'])[perm]
                train_targets = np.array(train_data['label'])[perm]
                batch_time = time.time()

                # train_steps = 40
                train_steps = train_length // self.para.batch_size

                for train_step in range(train_steps):
                    begin = train_step * self.para.batch_size
                    end = (train_step + 1) * self.para.batch_size
                    batch_s1, batch_s1_len = self.get_batch(train_s1[begin: end])
                    batch_s2, batch_s2_len = self.get_batch(train_s2[begin: end])
                    batch_labels = train_targets[begin : end]
    
                    train_dict = {
                        self.s1_embedded: batch_s1,
                        self.s1_lengths: batch_s1_len, 
                        self.s2_embedded: batch_s2,
                        self.s2_lengths: batch_s2_len, 
                        self.labels: batch_labels.T,
                        self.eta: self.learning_rate}

                    _, batch_loss, current_step, batch_summary = self.sess.run(
                        [self.opt_op, self.loss, self.global_step, self.train_summary], 
                        feed_dict=train_dict)

                    print('\rStep %d loss: %0.5f' % (current_step, batch_loss), end='   ')
                    self.train_loss_writer.add_summary(batch_summary, current_step)
                    train_loss += batch_loss/self.para.eval_step

                    if current_step % self.para.eval_step == 0:

                        print("\nAverage training loss at epoch %d step %d:" % (epoch, current_step), train_loss)
                        dev_loss = 0
                        train_loss = 0 
                        perm = np.random.permutation(dev_length)
                        dev_s1 = np.array(dev_data['s1'])[perm]
                        dev_s2 = np.array(dev_data['s2'])[perm]
                        dev_targets = np.array(dev_data['label'])[perm]
                        
                        dev_steps = 30
                        # dev_steps = dev_length // self.para.batch_size

                        for dev_step in range(dev_steps):
                            begin = dev_step * self.para.batch_size
                            end = (dev_step + 1) * self.para.batch_size
                            batch_s1, batch_s1_len = self.get_batch(dev_s1[begin: end])
                            batch_s2, batch_s2_len = self.get_batch(dev_s2[begin: end])
                            batch_labels = dev_targets[begin : end]
            
                            dev_dict = {
                                self.s1_embedded: batch_s1,
                                self.s1_lengths: batch_s1_len, 
                                self.s2_embedded: batch_s2,
                                self.s2_lengths: batch_s2_len, 
                                self.labels: batch_labels.T}

                            batch_loss, dev_loss_summary, batch_accuracy = self.sess.run(
                                [self.loss, self.dev_loss_summary, self.accuracy], 
                                feed_dict=dev_dict)
                            dev_loss += batch_loss/dev_steps

                        self.dev_loss_writer.add_summary(dev_loss_summary, current_step)
                        # self.dev_accuracies.append(dev_loss)
                        print("\nAverage (across %d data points) dev loss at epoch %d step %d:" % 
                            (self.para.batch_size * dev_steps, epoch, current_step), dev_loss)
                        print('Learning rate: %0.6f' % self.learning_rate)
                        print('Accuracy: %0.2f' % batch_accuracy)
                        end_time = time.time()
                        print('Time for %d steps: %0.2f seconds' % (self.para.eval_step, end_time - batch_time))
                        batch_time = time.time()
                        secs = time.time() - self.start_time
                        hours = secs//3600
                        minutes = secs / 60 - hours * 60
                        print('Time elapsed: %d:%02d hours' % (hours, minutes))
                        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
                    if current_step % self.para.save_every == 0:
                        self.save_model(self.path + '/saved_models/', current_step)

                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print('End of Epoch %d' % epoch)
                dev_accuracy = 0
                dev_loss = 0
                # dev_steps = 5
                dev_steps = dev_length // self.para.batch_size
                for dev_step in range(dev_steps):
                    begin = dev_step * self.para.batch_size
                    end = (dev_step + 1) * self.para.batch_size
                    batch_s1, batch_s1_len = self.get_batch(dev_s1[begin: end])
                    batch_s2, batch_s2_len = self.get_batch(dev_s2[begin: end])
                    batch_labels = dev_targets[begin : end]
    
                    dev_dict = {
                        self.s1_embedded: batch_s1,
                        self.s1_lengths: batch_s1_len, 
                        self.s2_embedded: batch_s2,
                        self.s2_lengths: batch_s2_len, 
                        self.labels: batch_labels.T}

                    batch_accuracy, batch_loss, dev_accuracy_summary = self.sess.run(
                        [self.accuracy, self.loss, self.dev_accuracy_summary], 
                        feed_dict=dev_dict)
                    dev_accuracy += batch_accuracy/dev_steps
                    dev_loss += batch_loss/dev_steps

                self.dev_accuracy_writer.add_summary(dev_accuracy_summary, epoch)
                self.dev_accuracies.append(dev_accuracy)
                self.dev_loss.append(dev_loss)
                np.save('dev_accuracies.npy', np.array(self.dev_accuracies))
                np.save('dev_loss.npy', np.array(self.dev_loss))
                print('Current dev accuracy: %0.3f, Previous best dev accuracy: %0.3f' % (dev_accuracy, self.dev_accuracy))
                if (dev_accuracy > self.dev_accuracy) & (epoch > 0):
                    self.learning_rate = self.learning_rate/self.para.lrshrink
                    self.dev_accuracy = dev_accuracy
                    print('Dev accuracy improved, new learning rate: %0.6f' % self.learning_rate)
                else:
                    self.learning_rate = self.learning_rate*self.para.decay
                    print('Dev accuracy didnt improve, new learning rate: %0.6f' % self.learning_rate)
                
                self.save_model(self.path + '/saved_models/', self.global_step.eval(session = self.sess))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

        except KeyboardInterrupt:
            save = input('save?')
            if 'y' in save:
                self.save_model(self.path + '/saved_models/', self.global_step.eval(session = self.sess))

def make_paras(path):
    if not os.path.exists(path):
        os.makedirs(path)
    paras = Infersent_para(embedding_size = 620, 
        hidden_size = 2048, 
        hidden_layers = 1,
        batch_size = 64, 
        keep_prob_dropout = 1.0, 
        learning_rate = 0.1, 
        bidirectional = True,
        decay = 0.99,
        lrshrink = 5,
        eval_step = 500,
        uniform_init_scale = 0.1,
        clip_gradient_norm=5.0,
        save_every=1000000,
        epochs = 100)
    with open(path + 'paras.pkl', 'wb') as f:
        pkl.dump(paras, f)
    return paras

if __name__ == '__main__':
    path = '../dataset/SNLI/'
    GLOVE_PATH = "../dataset/GloVe/glove.840B.300d.txt"
    SKIPTHOUGHT_PATH = '/cluster/project2/mr/vetterle/skipthought/toronto_n5/'
    # SKIPTHOUGHT_PATH = "/Users/Jonas/Documents/Repositories/skipthought/models/toronto_n5/"
    output_path = '../training_data/'
    model_path = '../models/m9/'

    train, dev, test = get_nli(path)

    # word_vec = build_vocab(train['s1'] + train['s2'] + dev['s1'] + dev['s2'] + test['s1'] + test['s2'], GLOVE_PATH)
    word_vec = build_vocab(train['s1'] + train['s2'] + dev['s1'] + dev['s2'] + test['s1'] + test['s2'], SKIPTHOUGHT_PATH, skipthought=True)
    print(word_vec['<s>'].shape)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(model_path + 'vocab.pkl', 'wb') as f:
        pkl.dump(word_vec, f)

    for split in ['s1', 's2']:
        for data_type in ['train', 'dev', 'test']:
            eval(data_type)[split] = np.array([['<s>'] + [word for word in sent.split() if word in word_vec] +\
                                          ['</s>'] for sent in eval(data_type)[split]])     

    tf.reset_default_graph()
    paras = make_paras(model_path)
    if os.path.exists(model_path +'tensorboard'):
        print('Remove existing tensorboard folder')
        shutil.rmtree(model_path +'tensorboard')
    model = Infersent_model(vocab = word_vec, parameters = paras, path = model_path)
    model.sess = tf.Session()
    tf.global_variables_initializer().run(session = model.sess)
    model.initialise()
    model.train(train, dev)







