import sys
try:
    sys.path.remove('/usr/local/lib/python2.7/site-packages')
    sys.path.remove('/usr/local/Cellar/matplotlib/1.5.1/libexec/lib/python2.7/site-packages')
    sys.path.remove('/usr/local/Cellar/numpy/1.12.0/libexec/nose/lib/python2.7/site-packages')
except:
    next

import tensorflow as tf
import numpy as np
import pandas as pd
import os
import tasks.predict_dep as predict_dep
import tasks.predict_length as predict_length
import tasks.predict_words as predict_words
import encoder
import matplotlib.pyplot as plt 
import seaborn as sn

cluster = False

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
    # SKIPTHOUGHT_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/models/toronto_n5/'
    # MODEL_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/models/m6/'
    MODEL_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/models/toronto_n5/'

    INFERSENT_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/code/'
    SICK_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/eval/SICK/'
    SNLI_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/dataset/SNLI/'
    TORONTO_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/corpus/'
    SAVE_PATH = '..'

# sys.path.append('./')
# sys.path.append(SKIPTHOUGHT_PATH)
# sys.path.append(INFERSENT_PATH)
# from infersent.data import get_nli
# import skipthought
from skipthought.skipthought import Skipthought_para


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

MODELS = ['skipthought', 'infersent']
TASKS = ['Predict_words', 'Predict_length', 'Predict_dep']

MODEL = MODELS[0]
TASK = TASKS[1]
CBOW = False
UNTRAINED = False

_learning_rate = 0.0001
_batch_size = 64
_epochs = 10
_dropout = 0.9
        

if __name__ == '__main__':

    if TASK=='Predict_length':

        CBOW = False

        SAVE_PATH = './tasks/saved_models/{}/{}/{}'.format(MODEL, TASK, 'CBOW' if CBOW else 'noCBOW')

        tf.reset_default_graph()
        encoder = encoder.Encoder(
            model_name = MODEL, 
            model_path = MODEL_PATH, 
            cbow = CBOW,
            snli_path = SNLI_PATH)
        train, dev, test = predict_length.setup(
            snli_path = SNLI_PATH,
            toy = False)
        task = predict_length.Predict_length(
            encoder = encoder,
            learning_rate = _learning_rate,
            epochs=10)
        task.train_model(train, dev, y_train = None, y_dev = None, save_path = SAVE_PATH)
        test_labels = None
       
        confusion_matrix, df_accuracy,_ = task.test_model(test, test_labels, 1,
            saved_model_path =  SAVE_PATH, mclasses = None)

    if TASK=='Predict_words':

        CBOW = False
        UNTRAINED = False
        SAVE_PATH = './tasks/saved_models/{}/{}/{}{}'.format(
            MODEL, TASK, 'CBOW' if CBOW else 'noCBOW', 'UNTRAINED' if UNTRAINED else 'TRAINED')

        tf.reset_default_graph()
        encoder = encoder.Encoder(
            model_name = MODEL, 
            model_path = MODEL_PATH, 
            snli_path = SNLI_PATH,
            untrained = UNTRAINED, 
            cbow = CBOW)
        train, dev, test, pos, neg = predict_words.setup(
            snli_path = SNLI_PATH,
            toy = True)
        task = predict_words.Predict_words(
            learning_rate = _learning_rate,
            batch_size = _batch_size, 
            encoder = encoder)

        task.train_model(train, dev, [pos[0], neg[0]], [pos[1], neg[1]], save_path = SAVE_PATH)

        confusion_matrix, df_accuracy,_ = task.test_model(test, [pos[2], neg[2]], 1,
            saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = None)


    elif TASK=='Predict_dep':

        CBOW = False
        UNTRAINED = False
        n_iter = 2
        m_iter = 3
        SAVE_PATH = './tasks/saved_models/{}/{}/{}{}{}'.format(
            MODEL, TASK, n_iter, 'CBOW' if CBOW else 'noCBOW', 'UNTRAINED' if UNTRAINED else 'TRAINED')

        tf.reset_default_graph()
        encoder = encoder.Encoder(
            model_name = MODEL, 
            model_path = MODEL_PATH, 
            snli_path = SNLI_PATH,
            untrained = UNTRAINED, 
            cbow = CBOW)
        train, dev, train_labels, dev_labels, all_relations_list, mclasses = predict_dep.setup(
            n_iter = n_iter, 
            m_iter = m_iter, 
            vocab = encoder.model.vocab, 
            snli_path = SNLI_PATH,
            toy = True)
        task = predict_dep.Predict_dep(
            dependency_list = all_relations_list, 
            learning_rate = _learning_rate, 
            keep_prob = _dropout, 
            batch_size = _batch_size, 
            use_sent = True, 
            encoder = encoder)

        task.train_model(train, dev, train_labels, dev_labels, save_path = SAVE_PATH)
        confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
            saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = None)
        confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
            saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = mclasses)

        CBOW = True
        UNTRAINED = False
        
        task = Predict_dep(vocab, dependency_list = all_relations_list, sent_dim = SENT_DIM, word_dim = WORD_DIM,
            learning_rate = _learning_rate, keep_prob = _dropout, batch_size = _batch_size, use_sent = True, embeddings = embeddings)
        train_model(train, dev, train_labels, dev_labels)
        confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
            saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = None)
        confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
            saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = mclasses)

        CBOW = False
        UNTRAINED = True
        
        task = Predict_dep(vocab, dependency_list = all_relations_list, sent_dim = SENT_DIM, word_dim = WORD_DIM,
            learning_rate = _learning_rate, keep_prob = _dropout, batch_size = _batch_size, use_sent = True, embeddings = embeddings)
        train_model(train, dev, train_labels, dev_labels)
        confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
            saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = None)
        confusion_matrix, df_accuracy,_ = test_model(test, test_labels, 1,
            saved_model_path = '%s/%s/%s/sent_words%d%s%s' % (SAVE_PATH, MODEL, TASK, n_iter, CBOW, UNTRAINED), mclasses = mclasses)

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