import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
import os

class Predict_dep(object):

    def __init__(self, 
        vocab, 
        dependency_list,
        sent_dim = 4096,
        word_dim = 300,
        learning_rate = 0.001,
        keep_prob = 1.0,
        decay_steps = 1000,
        decay = 1.0,
        batch_size = 32,
        use_sent = True,
        embeddings = None):

        self.vocab = vocab
        self.vocab_size = len(self.vocab)

        # self.dependency_list = ['root', 'dep','aux','auxpass','cop','arg','agent','comp','acomp','ccomp','xcomp','obj','dobj','iobj','pobj','subj','nsubj',
        # 'csubj','csubjpass','cc','conj','expl','mod','amod','appos','advcl','det','predet','preconj','vmod','mwe','mark','advmod','neg','rcmod',
        # 'quantmod','nn','npadvmod','tmod','num','number','prep','poss','possessive','prt','parataxis','goeswith','punct','ref','sdep','xsubj', 'nmod', 'case', 
        # 'acl', 'compound', 'discourse','nummod', 'ROOT','nsubjpass']
        self.labels_list = sorted(dependency_list)
        self.labels_dict = {}
        for i,rel in enumerate(sorted(self.labels_list)):
            self.labels_dict[rel] = i

        self.sent_dim = sent_dim
        self.word_dim = word_dim
        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.decay_steps = decay_steps
        self.decay = decay
        self.batch_size = batch_size
        self.use_sent = use_sent
        self.embeddings = embeddings

        self.graph = tf.get_default_graph()
        self.initializer = tf.random_normal_initializer()

        self.X = tf.placeholder(
            tf.float32, 
            [None, self.sent_dim], 
            'sentences')
        self.words = tf.placeholder(
            tf.float32, 
            [None, self.word_dim * 2], 
            'words')
        self.y = tf.placeholder(
            tf.int32, 
            [None], 
            'labels')
        self.do = tf.placeholder(tf.float32, name='dropout')

        self.global_step = tf.Variable(0, 
            name = 'global_step', 
            trainable = False)

        if self.use_sent:
            self.features = tf.concat([self.X, self.words], axis = 1)
        else:
            self.features = tf.concat([self.words], axis = 1)

        self.hidden = tf.nn.dropout(tf.contrib.layers.fully_connected(
            self.features, 100), keep_prob = self.do)

        self.logits = tf.contrib.layers.fully_connected(
            self.hidden, len(self.labels_list), activation_fn=None)

        self.prediction = tf.argmax(tf.nn.softmax(self.logits),1)
        self.l = tf.nn.sparse_softmax_cross_entropy_with_logits(
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

        parse = [[list(sent.triples()) for sent in parse] for parse in parser.raw_parse_sents(sentences)]
        return labels

    def get_accuracy(self, data, labels, prediction):

        correct = np.sum(labels==prediction)
        total = len(labels)
        accuracy = correct/total
        return accuracy

    def get_f1_pre_rec(self, labels, prediction):

        pre, rec, f1, _ = precision_recall_fscore_support(
            y_true = labels, 
            y_pred = prediction,
            labels = [self.labels_dict[i] for i in self.labels_list])

        counts = np.zeros([len(self.labels_list), 1])
        for i in labels:
            counts[i] += 1

        return np.expand_dims(pre,1), np.expand_dims(rec,1), np.expand_dims(f1,1), counts

    def get_auc(self, labels, prediction):

        return roc_auc_score(labels, prediction)


    def get_confusion(self, labels, prediction):

        matrix = confusion_matrix(labels, prediction, 
            labels = [self.labels_dict[i] for i in self.labels_list])
        return matrix

    def get_embedding(self, word):

        if self.embeddings == None:
            return self.vocab[word]
        else:
            return self.embeddings[self.vocab[word],:]

    def run_batch(self, data, string_sentences, list_of_labels, train = False, mclasses = None):
        
        sentences = []
        words = []
        labels = []

        if mclasses == None:
            for i, example in enumerate(list_of_labels):
                for triplet in example:
                    try:
                        gov = self.vocab[triplet[1]]
                        dep = self.vocab[triplet[0]]
                        feat = np.concatenate((dep, gov))
                        sentences.append(data[0][i])
                        words.append(feat)
                        labels.append(self.labels_dict[triplet[2].split(':')[0]])
                    except:
                        next
        else:
            for i, example in enumerate(list_of_labels):
                for triplet in example:
                    try:
                        
                        if mclasses[triplet[0]] != triplet[2]:
                            gov = self.vocab[triplet[1]]
                            dep = self.vocab[triplet[0]]
                            feat = np.concatenate((dep, gov))
                            sentences.append(data[0][i])
                            words.append(feat)
                            labels.append(self.labels_dict[triplet[2].split(':')[0]])

                    except:
                        next

        sentences = np.array(sentences)
        labels = np.array(labels)

        if train:
            feed_dict = {self.X: sentences,
                    self.words: words,
                      self.y: labels,
                      self.do: self.keep_prob}
            _, batch_loss = self.sess.run(
                [self.opt_op, self.loss], 
                feed_dict=feed_dict)
            return batch_loss

        else:
            feed_dict = {self.X: sentences,
                    self.words: words,
                      self.y: labels,
                      self.do: 1.0}
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
        print('Dependency tag model restored')
