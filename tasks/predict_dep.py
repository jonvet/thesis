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

class Predict_dep(object):

    def __init__(self, 
        encoder,
        dependency_list,
        learning_rate = 0.001,
        keep_prob = 1.0,
        decay_steps = 1000,
        decay = 1.0,
        batch_size = 32,
        epochs = 10,
        use_sent = True,
        embeddings = None):

        self.vocab = encoder.model.vocab
        self.vocab_size = len(self.vocab)
        self.encoder = encoder

        self.labels_list = sorted(dependency_list)
        self.labels_dict = {}
        for i,rel in enumerate(sorted(self.labels_list)):
            self.labels_dict[rel] = i

        self.sent_dim = encoder.sent_dim
        self.word_dim = encoder.word_dim

        self.learning_rate = learning_rate
        self.keep_prob = keep_prob
        self.decay_steps = decay_steps
        self.decay = decay
        self.batch_size = batch_size
        self.epochs = epochs
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


    def train_model(self, X_train, X_dev, y_train, y_dev, save_path):
        self.sess = tf.Session(graph = self.graph)
        tf.global_variables_initializer().run(session = self.sess)

        best_f1 = 0
        for epoch in range(self.epochs):
            print('\nStarting epoch %d' % epoch)
            perm = np.random.permutation(len(X_train))
            train_perm = X_train[perm]

            train_labels_perm = np.array(y_train)[perm]
            avg_loss = 0
            steps = len(X_train) // self.batch_size

            for step in range(0, len(X_train), self.batch_size):

                sentences = train_perm[step:(step+self.batch_size)]
                sentence_data = self.encoder.embed(sentences)
                labels = train_labels_perm[step:(step+self.batch_size)]

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
            labels = y[step:(step+self.batch_size)]
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


def setup(n_iter, m_iter, vocab, snli_path, toy=False):

    print('Loading corpus')
    train, dev, test = get_nli(snli_path)
    train = np.array(train['s2'])
    dev = np.array(dev['s2'])
    test = np.array(test['s2'])

    with open ('./tasks/preprocess_data/train_s2.pkl', 'rb') as f:
        train_labels = pkl.load(f)
    with open ('./tasks/preprocess_data/dev_s2.pkl', 'rb') as f:
        dev_labels = pkl.load(f)
    with open ('./tasks/preprocess_data/test_s2.pkl', 'rb') as f:
        test_labels = pkl.load(f)

    if toy:
        train, train_labels = train[:500], train_labels[:500]
        dev, dev_labels = dev[:500], dev_labels[:500]
        test, test_labels = test[:500], test_labels[:500]

    train, train_labels = remove_short_sentences(train, train_labels)
    dev, dev_labels = remove_short_sentences(dev, dev_labels)
    test, test_labels = remove_short_sentences(test, test_labels)

    train_labels = remove_relations(train_labels, ['ROOT', 'root', 'punct'])
    dev_labels = remove_relations(dev_labels, ['ROOT', 'root', 'punct'])
    test_labels = remove_relations(test_labels, ['ROOT', 'root', 'punct'])

    train_labels = get_none_class(train, train_labels, num_samples=n_iter, max_iter=m_iter)
    dev_labels = get_none_class(dev, dev_labels, num_samples=n_iter, max_iter=m_iter)
    test_labels = get_none_class(test, test_labels, num_samples=n_iter, max_iter=m_iter)

    all_relations_dict = get_all_relations([train_labels, dev_labels, test_labels])
    all_relations_list = list(set([rel.split(':')[0] for rel in all_relations_dict.keys()] + ['None']))
    get_rel_stats(all_relations_dict, [train, dev, test])

    assert len(train) == len(train_labels)
    assert len(dev) == len(dev_labels)
    assert len(test) == len(test_labels)
    all_classes, mclasses = get_majority_class(vocab, test_labels)

    return train, dev, train_labels, dev_labels, all_relations_list, mclasses


def get_all_relations(data):

    all_labels = defaultdict(int)
    for part in data:
        for sentence in part:
            for triplet in sentence:
                all_labels[triplet[2]] += 1

    return all_labels

def get_rel_stats(dict, data):

    df = pd.DataFrame.from_dict(dict, orient='index')

    print('Number of occurences of each relation type by number of sentences:')
    print(df/np.sum([len(i) for i in data]))
    print('Average number of relations per sentence: %0.2f' % (df.sum()/np.sum([len(i) for i in data])))
    print('Total number of occurences of each relation:')
    print(df)

def remove_relations(labels, relations):

    new_labels = []
    for sentence in labels:
        sent = []
        for triplet in sentence:
            if triplet[2] not in relations:
                sent.append(triplet)
        new_labels.append(sent)
    return new_labels

def get_none_class(sent, lab, num_samples=3, max_iter=10):

    new_labels = []
    for sentence, label in zip(sent,lab):
        i, num = 0, 0
        new_label = label
        while (i<max_iter) & (num<num_samples):
            try:
                sample = random.sample(sentence.split(' ')[:-1], 2)
                rels = [[l[0], l[1]] for l in label]
                # if (sample not in rels) and (sample not in rels[::-1]):
                if sample not in rels:
                    new_rel = sample + ['None']
                    new_label.append(new_rel)
                    num += 1
            except:
                print(sentence)
            i += 1
        new_labels.append(new_label)

    return new_labels

def get_majority_class(vocab, data):

    all_classes = defaultdict(lambda : defaultdict(int))
    for word in vocab.keys():
        all_classes[word] = defaultdict(int)
    for sentence in data:
        for triplet in sentence:
            if triplet[2] != 'None':
                all_classes[triplet[0]][triplet[2]] += 1
    mclasses = {}
    for word, classes in all_classes.items():
        m_class = ''
        m_count = 0
        for cl, co in classes.items():
            if co>m_count:
                m_class = cl
                m_count = co
        if m_count>0:
            mclasses[word] = m_class
    return all_classes, mclasses

def remove_short_sentences(sentences, labels=None, min_len=4):

    lengths = [len(sentence.split()) for sentence in sentences]
    sentences = np.array(sentences)[np.array(lengths) > min_len]
    if labels == None:
        return sentences
    else:
        labels = np.array(labels)[np.array(lengths) > min_len]
        return sentences, labels