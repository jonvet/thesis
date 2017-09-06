import sys
try:
    sys.path.remove('/usr/local/lib/python2.7/site-packages')
    sys.path.remove('/usr/local/Cellar/matplotlib/1.5.1/libexec/lib/python2.7/site-packages')
    sys.path.remove('/usr/local/Cellar/numpy/1.12.0/libexec/nose/lib/python2.7/site-packages')
except:
    next

import tensorflow as tf
import numpy as np
import pickle as pkl 
from collections import defaultdict
import pandas as pd
import os
from predict_words import Predict_words
from predict_words2 import Predict_words2
from predict_length import Predict_length
from predict_dep import Predict_dep
from sklearn.model_selection import StratifiedShuffleSplit
import time
import datetime
import matplotlib.pyplot as plt
import seaborn as sn
import random

cluster = True

if cluster:
    SKIPTHOUGHT_PATH = '/home/vetterle/skipthought/code/'
    # MODEL_PATH = '/cluster/project2/mr/vetterle/infersent/m6/'
    MODEL_PATH = '/cluster/project2/mr/vetterle/skipthought/toronto_n5/'

    SKIPTHOUGHT_PATH = '/cluster/project2/mr/vetterle/skipthought/toronto_n5/'
    INFERSENT_PATH = '/home/vetterle/InferSent/code/'
    SICK_PATH = '/home/vetterle/skipthought/eval/SICK/'
    SNLI_PATH = '/home/vetterle/InferSent/dataset/SNLI/'
    TORONTO_PATH = '/cluster/project6/mr_corpora/vetterle/toronto/'
    SAVE_PATH = '/cluster/project2/mr/vetterle/thesis'

else:
    # SKIPTHOUGHT_PATH = '/Users/Jonas/Documents/Repositories/skipthought/code/'
    SKIPTHOUGHT_PATH = '/Users/Jonas/Documents/Repositories/skipthought/models/toronto_n5/'
    MODEL_PATH = '/Users/Jonas/Documents/Repositories/InferSent/models/m4/'
    # MODEL_PATH = '/Users/Jonas/Documents/Repositories/skipthought/models/toronto_n5/'

    INFERSENT_PATH = '/Users/Jonas/Documents/Repositories/InferSent/code/'
    SICK_PATH = '/Users/Jonas/Documents/Repositories/skipthought/eval/SICK/'
    SNLI_PATH = '/Users/Jonas/Documents/Repositories/InferSent/dataset/SNLI/'
    TORONTO_PATH = '/Users/Jonas/Documents/Repositories/skipthought/corpus/'
    SAVE_PATH = '..'

sys.path.append(SKIPTHOUGHT_PATH)
sys.path.append(INFERSENT_PATH)
from data import get_nli

MODELS = ['skipthought', 'infersent']
TASKS = ['Predict_words', 'Predict_length', 'Predict_dep', 'Predict_words2']

MODEL = MODELS[0]
TASK = TASKS[2]
CBOW = False
UNTRAINED = False

_learning_rate = 0.0001
_batch_size = 64
_epochs = 10
_dropout = 0.9

def embed(sentences):

    sentences = [[w for w in sentence.split(' ')] for sentence in sentences]
    sentences_embedded, sentences_lengths = [], []

    if MODEL == 'skipthought':
        sentences_lengths = np.array([len(sentence) for sentence in sentences], dtype=np.int32)
        batch_embedded = np.full([len(sentences), np.max(sentences_lengths), 620], vocab['<PAD>'], dtype=np.float32)
        for i, sentence in enumerate(sentences):
            words = [vocab[word] if word in vocab else vocab['<OOV>'] for word in sentence]
            batch_embedded[i, :len(sentence), :] = np.array(words)
        if CBOW:
            sentences_embedded = np.mean(batch_embedded, axis = 1)
        else:
            test_dict = {model.sentences_embedded: batch_embedded, 
            model.sentences_lengths: sentences_lengths}
            sentences_embedded = model.encoded_sentences.eval(session = model.sess, feed_dict=test_dict)
        return (sentences_embedded, sentences_lengths)

    else:
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in vocab]
            if not s_f:
                s_f = ['</s>']
            sentences[i] = s_f
        if CBOW:
            batch_words = [np.array([vocab[word] for word in sentence]) for sentence in sentences]
            batch_embedded = [np.mean(sentence, axis = 0) for sentence in batch_words]
            batch_lengths = [len(sentence) for sentence in sentences]
            sentences_embedded.append(batch_embedded)
            sentences_lengths.append(batch_lengths)

        else:
            batch_s, batch_l = model.get_batch(sentences)
            test_dict = {
                model.s1_embedded: batch_s,
                model.s1_lengths: batch_l}
            batch_embedded = model.sess.run(
                model.s1_states_h, 
                feed_dict=test_dict)
            sentences_embedded.append(batch_embedded)
            sentences_lengths.append(batch_l)
        return (np.squeeze(sentences_embedded), np.squeeze(sentences_lengths))

def get_all_relations(data):

    all_labels = defaultdict(int)
    for part in data:
        for sentence in part:
            for triplet in sentence:
                all_labels[triplet[2]] += 1

    return all_labels

def get_rel_stats(dict):

    df = pd.DataFrame.from_dict(dict, orient='index')

    print('Number of occurences of each relation type by number of sentences:')
    print(df/(len(train)+len(dev)+len(test)))
    print('Average number of relations per sentence: %0.2f' % (df.sum()/(len(train)+len(dev)+len(test))))
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

def test_model(X, y, step = None, saved_model_path = None, mclasses = None):

    if saved_model_path != None:
        task.sess = tf.Session(graph = task.graph)
        task.load_model(saved_model_path, step)

    num_classes = len(task.labels_list)
    dev_loss, dev_accuracy = 0, 0
    dev_f1, dev_pre, dev_rec, dev_counts = np.zeros([num_classes, 1]), np.zeros([num_classes, 1]), np.zeros([num_classes, 1]), np.zeros([num_classes, 1])
    dev_confusion = np.zeros([num_classes, num_classes])
    dev_steps = len(X) // _batch_size

    all_labels, all_preds = [], []

    for step in range(0, len(X), _batch_size):

        print('\rStep %d/%d' % (step/_batch_size, dev_steps), end = '    ')

        sentences = X[step:(step+_batch_size)]
        sentence_data = embed(sentences)

        if TASK == 'Predict_dep':
            labels = y[step:(step+_batch_size)]
        elif TASK == 'Predict_length':
            labels = task.get_labels(sentence_data, sentences)
        elif TASK == 'Predict_words2':
            labels = [y[0][step:(step+_batch_size)], y[1][step:(step+_batch_size)]]

        prediction, loss, labels = task.run_batch(sentence_data, sentences, labels, train=False, mclasses=mclasses)

        accuracy = task.get_accuracy(sentence_data, labels, prediction)
        dev_accuracy += accuracy/dev_steps

        all_labels += list(labels)
        all_preds += list(prediction)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    dev_pre, dev_rec, dev_f1, dev_counts = task.get_f1_pre_rec(all_labels, all_preds)
    matrix = task.get_confusion(all_labels, all_preds)

    print('Test accuracy: %0.4f\n' % dev_accuracy)
    weights = dev_counts/np.sum(dev_counts)
    weights_no_none = dev_counts[1:]/np.sum(dev_counts[1:])
    weighted_f1 = float(np.matmul(weights.T,dev_f1))
    weighted_f1_no_none = float(np.matmul(weights_no_none.T,dev_f1[1:]))
    print('Weighted F1 score: %0.4f\n' % weighted_f1)
    print('Weighted F1 score (no none): %0.4f\n' % weighted_f1_no_none)

    df_accuracy = pd.DataFrame(data=np.concatenate((dev_f1, dev_pre, dev_rec, dev_counts), axis=1),
        columns=['F1', 'Precision', 'Recall', 'n'], index=task.labels_list)
    print(df_accuracy)
    # return dev_confusion, df_accuracy, weighted_f1
    return dev_confusion, df_accuracy, weighted_f1_no_none

def train_model(X_train, X_dev, y_train, y_dev):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7
    task.sess = tf.Session(graph = task.graph, config=config)
    tf.global_variables_initializer().run(session = task.sess)

    best_f1 = 0
    for epoch in range(_epochs):
        print('\nStarting epoch %d' % epoch)
        perm = np.random.permutation(len(X_train))
        train_perm = X_train[perm]

        if TASK == 'Predict_dep':
            train_labels_perm = np.array(y_train)[perm]
        elif TASK == 'Predict_length':
            y_dev = None
        elif TASK == 'Predict_words2':
            train_labels_perm = [np.array(y_train[0])[perm], np.array(y_train[1])[perm]]

        avg_loss = 0
        dev_loss = 0
        dev_accuracy = 0

        steps = len(X_train) // _batch_size

        for step in range(0, len(X_train), _batch_size):

            sentences = train_perm[step:(step+_batch_size)]
            sentence_data = embed(sentences)
            
            if TASK == 'Predict_dep':
                labels = train_labels_perm[step:(step+_batch_size)]
            elif TASK == 'Predict_length':
                labels = task.get_labels(sentence_data, sentences)
            elif TASK == 'Predict_words2':
                labels = [train_labels_perm[0][step:(step+_batch_size)], train_labels_perm[1][step:(step+_batch_size)]]
            loss = task.run_batch(sentence_data, sentences, labels, train=True)
            avg_loss += loss/steps
            print('\rBatch loss at step %d: %0.5f' % (step/_batch_size, loss), end = '    ')
           
        _,_,f1 = test_model(X_dev, y_dev)

        if f1>best_f1:
            save_path = '%s/%s/%s/sent_words%d%s%s/' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED)
            task.save_model(save_path, 1)   
        else:
            break
        

# if __name__ == '__main__':

print('Loading corpus')
train, dev, test = get_nli(SNLI_PATH)
train = np.array(train['s2'])
dev = np.array(dev['s2'])
test = np.array(test['s2'])

print('Loading saved model')
tf.reset_default_graph()
embeddings = None # in case of 'cbow' or 'infersent' model
n_iter = 0

with open(MODEL_PATH + 'vocab.pkl', 'rb') as f:
    vocab = pkl.load(f)

if MODEL == 'skipthought':
    from skipthought import Skipthought_para
    from skipthought import Skipthought_model
    step = 488268
    with open(MODEL_PATH + 'paras.pkl', 'rb') as f:
        paras = pkl.load(f)
    model = Skipthought_model(vocab = vocab, parameters = paras, path = MODEL_PATH)
    if UNTRAINED:
        model.sess = tf.Session(graph = model.graph)
        tf.global_variables_initializer().run(session = model.sess)
    else:
        model.load_model(MODEL_PATH, step)

    dictionary = defaultdict(int)
    embeddings = np.load(MODEL_PATH + 'expanded_embeddings.npy')
    with open(MODEL_PATH + 'expanded_vocab.pkl', 'rb') as f:
        expanded_vocab = pkl.load(f)

    for part in [train, dev, test]:
        for i in range(len(part)):
            sentence = part[i].split()
            for word in sentence:
                dictionary[word] += 1
    vocab = {}
    for word in ['<OOV>','<PAD>']:
        vocab[word] = embeddings[expanded_vocab[word]]
    for word in dictionary.keys():
        try:
            vocab[word] = embeddings[expanded_vocab[word]]
        except:
            next
    print('Found {0}(/{1}) words with word2vec vectors'.format(len(vocab), len(expanded_vocab)))
    model.vocab = vocab
    SENT_DIM = 620 if CBOW else 2400
    WORD_DIM = 620
    embeddings = None

elif MODEL == 'infersent':
    from infersent import Infersent_para
    from infersent import Infersent_model
    # step = 77247
    step = 128745
    with open(MODEL_PATH + 'paras.pkl', 'rb') as f:
        paras = pkl.load(f)
    model = Infersent_model(vocab = vocab, parameters = paras, path = MODEL_PATH)
    if UNTRAINED:
        model.sess = tf.Session(graph = model.graph)
        tf.global_variables_initializer().run(session = model.sess)
    else:
        model.load_model(MODEL_PATH, step)
    model.para.batch_size = _batch_size
    SENT_DIM = 300 if CBOW else 4096
    WORD_DIM = 300

print('%s model loaded' %MODEL)

if TASK=='Predict_dep':
    with open ('../preprocess_data/train_s2.pkl', 'rb') as f:
        train_labels = pkl.load(f)
    with open ('../preprocess_data/dev_s2.pkl', 'rb') as f:
        dev_labels = pkl.load(f)
    with open ('../preprocess_data/test_s2.pkl', 'rb') as f:
        test_labels = pkl.load(f)

    train, train_labels = remove_short_sentences(train, train_labels)
    dev, dev_labels = remove_short_sentences(dev, dev_labels)
    test, test_labels = remove_short_sentences(test, test_labels)

    train_labels = remove_relations(train_labels, ['ROOT', 'root', 'punct'])
    dev_labels = remove_relations(dev_labels, ['ROOT', 'root', 'punct'])
    test_labels = remove_relations(test_labels, ['ROOT', 'root', 'punct'])

    # CBOW = False
    # UNTRAINED = False
    # task = Predict_dep(vocab, dependency_list = all_relations_list, sent_dim = SENT_DIM, word_dim = WORD_DIM,
    #     learning_rate = _learning_rate, keep_prob = _dropout, batch_size = _batch_size, use_sent = True, embeddings = embeddings)
    # train_model(train, dev, train_labels, dev_labels)
    # confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
    #     saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = None)
    # confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
    #     saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = mclasses)

    

    n_iter = 20
    m_iter = 200

    train_labels = get_none_class(train, train_labels, num_samples=n_iter, max_iter=m_iter)
    dev_labels = get_none_class(dev, dev_labels, num_samples=n_iter, max_iter=m_iter)
    test_labels = get_none_class(test, test_labels, num_samples=n_iter, max_iter=m_iter)

    all_relations_dict = get_all_relations([train_labels, dev_labels, test_labels])
    all_relations_list = list(set([rel.split(':')[0] for rel in all_relations_dict.keys()] + ['None']))
    get_rel_stats(all_relations_dict)

    assert len(train) == len(train_labels)
    assert len(dev) == len(dev_labels)
    assert len(test) == len(test_labels)
    all_classes, mclasses = get_majority_class(vocab, test_labels)

    CBOW = False
    UNTRAINED = False
    with open(MODEL_PATH + 'vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    tf.reset_default_graph()
    
    task = Predict_dep(vocab, dependency_list = all_relations_list, sent_dim = SENT_DIM, word_dim = WORD_DIM,
        learning_rate = _learning_rate, keep_prob = _dropout, batch_size = _batch_size, use_sent = True, embeddings = embeddings)
    train_model(train, dev, train_labels, dev_labels)
    confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
        saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = None)
    confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
        saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = mclasses)

    CBOW = True
    UNTRAINED = False
    with open(MODEL_PATH + 'vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    tf.reset_default_graph()
    if MODEL == 'skipthought':
        from skipthought import Skipthought_para
        from skipthought import Skipthought_model
        step = 488268
        with open(MODEL_PATH + 'paras.pkl', 'rb') as f:
            paras = pkl.load(f)
        model = Skipthought_model(vocab = vocab, parameters = paras, path = MODEL_PATH)
        if UNTRAINED:
            model.sess = tf.Session(graph = model.graph)
            tf.global_variables_initializer().run(session = model.sess)
        else:
            model.load_model(MODEL_PATH, step)

        dictionary = defaultdict(int)
        embeddings = np.load(MODEL_PATH + 'expanded_embeddings.npy')
        with open(MODEL_PATH + 'expanded_vocab.pkl', 'rb') as f:
            expanded_vocab = pkl.load(f)

        for part in [train, dev, test]:
            for i in range(len(part)):
                sentence = part[i].split()
                for word in sentence:
                    dictionary[word] += 1
        vocab = {}
        for word in ['<OOV>','<PAD>']:
            vocab[word] = embeddings[expanded_vocab[word]]
        for word in dictionary.keys():
            try:
                vocab[word] = embeddings[expanded_vocab[word]]
            except:
                next
        print('Found {0}(/{1}) words with word2vec vectors'.format(len(vocab), len(expanded_vocab)))
        model.vocab = vocab
        SENT_DIM = 620 if CBOW else 2400
        WORD_DIM = 620
        embeddings = None

    elif MODEL == 'infersent':
        from infersent import Infersent_para
        from infersent import Infersent_model
        # step = 77247
        step = 128745
        with open(MODEL_PATH + 'paras.pkl', 'rb') as f:
            paras = pkl.load(f)
        model = Infersent_model(vocab = vocab, parameters = paras, path = MODEL_PATH)
        if UNTRAINED:
            model.sess = tf.Session(graph = model.graph)
            tf.global_variables_initializer().run(session = model.sess)
        else:
            model.load_model(MODEL_PATH, step)
        model.para.batch_size = _batch_size
        SENT_DIM = 300 if CBOW else 4096
        WORD_DIM = 300
    task = Predict_dep(vocab, dependency_list = all_relations_list, sent_dim = SENT_DIM, word_dim = WORD_DIM,
        learning_rate = _learning_rate, keep_prob = _dropout, batch_size = _batch_size, use_sent = True, embeddings = embeddings)
    train_model(train, dev, train_labels, dev_labels)
    confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
        saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = None)
    confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
        saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = mclasses)

    CBOW = False
    UNTRAINED = True
    with open(MODEL_PATH + 'vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    tf.reset_default_graph()
    if MODEL == 'skipthought':
        from skipthought import Skipthought_para
        from skipthought import Skipthought_model
        step = 488268
        with open(MODEL_PATH + 'paras.pkl', 'rb') as f:
            paras = pkl.load(f)
        model = Skipthought_model(vocab = vocab, parameters = paras, path = MODEL_PATH)
        if UNTRAINED:
            model.sess = tf.Session(graph = model.graph)
            tf.global_variables_initializer().run(session = model.sess)
        else:
            model.load_model(MODEL_PATH, step)

        dictionary = defaultdict(int)
        embeddings = np.load(MODEL_PATH + 'expanded_embeddings.npy')
        with open(MODEL_PATH + 'expanded_vocab.pkl', 'rb') as f:
            expanded_vocab = pkl.load(f)

        for part in [train, dev, test]:
            for i in range(len(part)):
                sentence = part[i].split()
                for word in sentence:
                    dictionary[word] += 1
        vocab = {}
        for word in ['<OOV>','<PAD>']:
            vocab[word] = embeddings[expanded_vocab[word]]
        for word in dictionary.keys():
            try:
                vocab[word] = embeddings[expanded_vocab[word]]
            except:
                next
        print('Found {0}(/{1}) words with word2vec vectors'.format(len(vocab), len(expanded_vocab)))
        model.vocab = vocab
        SENT_DIM = 620 if CBOW else 2400
        WORD_DIM = 620
        embeddings = None

    elif MODEL == 'infersent':
        from infersent import Infersent_para
        from infersent import Infersent_model
        # step = 77247
        step = 128745
        with open(MODEL_PATH + 'paras.pkl', 'rb') as f:
            paras = pkl.load(f)
        model = Infersent_model(vocab = vocab, parameters = paras, path = MODEL_PATH)
        if UNTRAINED:
            model.sess = tf.Session(graph = model.graph)
            tf.global_variables_initializer().run(session = model.sess)
        else:
            model.load_model(MODEL_PATH, step)
        model.para.batch_size = _batch_size
        SENT_DIM = 300 if CBOW else 4096
        WORD_DIM = 300
    task = Predict_dep(vocab, dependency_list = all_relations_list, sent_dim = SENT_DIM, word_dim = WORD_DIM,
        learning_rate = _learning_rate, keep_prob = _dropout, batch_size = _batch_size, use_sent = True, embeddings = embeddings)
    train_model(train, dev, train_labels, dev_labels)
    confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
        saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = None)
    confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
        saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = mclasses)


elif TASK=='Predict_length':
    task = Predict_length(dim = SENT_DIM, learning_rate = _learning_rate)

    train_model(train, dev, y_train = None, y_dev = None)
    test_labels = None
   
    # test_model(test, test_labels)
    confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
        saved_model_path = '%s/%s/%s/sent_words0%s%s' % (SAVE_PATH, MODEL, TASK, CBOW), mclasses = None)

elif TASK=='Predict_words2':
    task = Predict_words2(vocab, sent_dim = SENT_DIM, word_dim = WORD_DIM, learning_rate = _learning_rate)
    # train = train[:500]
    # dev = dev[:500]
    # test = test[:500]

    train = remove_short_sentences(train)
    dev = remove_short_sentences(dev)
    test = remove_short_sentences(test)

    pos = task.get_pos_samples([train,dev,test])
    neg = task.get_neg_samples([train,dev, test], pos)

    train_model(train, dev, [pos[0], neg[0]], [pos[1], neg[1]])

    confusion_matrix, df_accuracy,_ = test_model(test, [pos[2], neg[2]], 1,
        saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = None)

   

# sentences = []
# with open(TORONTO_PATH + 'test.txt', 'r') as f:
#     for line in f:
#         sentences.append(line)
# data = np.array(sentences[:300000])
# train, dev = balance_data(data)
# dev = dev[:1000]
# print('data balanced')


# task = Predict_words(vocab)
# 
# task = eval('%s(vocab, batch_size = _batch_size)' % TASK)

quit()
df_cm = pd.DataFrame(confusion_matrix, index = task.labels_list, columns = task.labels_list)
df_cm_norm = df_cm/df_cm.sum(axis=0)
df_cm_norm = df_cm_norm.fillna(0)
sn.set(font_scale=0.7)
sn.heatmap(df_cm_norm)
plt.yticks(rotation=0)
plt.xticks(rotation=90)
# plt.show()
# plt.savefig('conf_matrix_all.png')
plt.savefig('conf_matrix_no_mclass.png')
plt.close()