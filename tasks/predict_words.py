import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import os
from infersent.data import get_nli
import pickle as pkl
import random
import pandas as pd

class Predict_words(object):

    def __init__(self, 
        encoder,
        learning_rate = 0.001,
        decay_steps = 1000,
        decay = 1.0,
        batch_size = 64,
        epochs = 10):

        self.vocab = encoder.model.vocab
        self.vocab_size = len(self.vocab)
        self.encoder = encoder

        self.sent_dim = encoder.sent_dim
        self.word_dim = encoder.word_dim
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay = decay
        self.batch_size = batch_size
        self.epochs = epochs
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
            self.features, 1, activation_fn=None, scope='output_layer')

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
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer'))
        saver.save(sess = self.sess, save_path = path + '/step_%d' % step)

    # def save_model(self, path, step):

    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     saver = tf.train.Saver()
    #     saver.save(sess = self.sess, save_path = path + '/step_%d' % step, write_state = False)

    def load_output_layer(self, path):

        self.sess = tf.Session(graph = self.graph)
        tf.global_variables_initializer().run(session = self.sess)

        with open(os.path.join(path, 'output_layer.pkl'), 'rb') as f:
            np_w, np_b = pkl.load(f)[0]
        with tf.variable_scope("output_layer", reuse=True):
            tf_w = tf.get_variable('weights')
            tf_b = tf.get_variable('biases')
        w_op = tf_w.assign(np_w)
        b_op = tf_b.assign(np_b)
        self.sess.run([w_op, b_op])

    def load_ft(self, path):
        with open(os.path.join(path, 'encoder_forget.pkl'), 'rb') as f:
            np_paras = pkl.load(f)
        encoder_vars = [t.name.split(':')[0] for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')]
        ops = []

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for a, b in zip(encoder_vars, np_paras):
                ops.append(tf.get_variable(a).assign(b))
        self.encoder.model.sess.run(ops)
        
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

            train_labels_perm = [np.array(y_train[0])[perm], np.array(y_train[1])[perm]]
            avg_loss = 0
            steps = len(X_train) // self.batch_size

            for step in range(0, len(X_train), self.batch_size):

                sentences = train_perm[step:(step+self.batch_size)]
                sentence_data = self.encoder.embed(sentences)
                labels = [train_labels_perm[0][step:(step+self.batch_size)], train_labels_perm[1][step:(step+self.batch_size)]]

                loss = self.run_batch(sentence_data, sentences, labels, train=True)
                avg_loss += loss/steps
                print('\rBatch loss at step %d: %0.5f' % (int(step/self.batch_size), loss), end = '    ')
               
            _,_,f1 = self.test_model(X_dev, y_dev)

            if f1>best_f1:
                # save_path = '%s/%s/%s/sent_words%d%s%s/' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED)
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

            print('\rStep {}/{}'.format(int(step/self.batch_size), dev_steps), end = '    ')
            sentences = X[step:(step+self.batch_size)]
            sentence_data = self.encoder.embed(sentences)
            labels = [y[0][step:(step+self.batch_size)], y[1][step:(step+self.batch_size)]]
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

def remove_short_sentences(sentences, labels=None, min_len=4):

    lengths = [len(sentence.split()) for sentence in sentences]
    sentences = np.array(sentences)[np.array(lengths) > min_len]
    if labels == None:
        return sentences
    else:
        labels = np.array(labels)[np.array(lengths) > min_len]
        return sentences, labels

def get_pos_samples(data):

    pos_samples= [[random.sample(sentence.split(' ')[:-1], 1)[0] for sentence in part] for part in data]
    return pos_samples

def get_neg_samples(data, pos_labels):

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
            part_list.append(candidate)
        neg_samples.append(part_list)    
    return neg_samples

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

    train = remove_short_sentences(train)
    dev = remove_short_sentences(dev)
    test = remove_short_sentences(test)

    pos = get_pos_samples([train,dev,test])
    neg = get_neg_samples([train,dev, test], pos)

    return train, dev, test, pos, neg