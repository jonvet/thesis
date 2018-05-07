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

cluster = True

if cluster:
    # MODEL_PATH = '/cluster/project2/mr/vetterle/infersent/m6/'
    MODEL_PATH = '/cluster/project2/mr/vetterle/skipthought/toronto_n5/'

    INFERSENT_PATH = '/home/vetterle/InferSent/code/'
    SICK_PATH = '/home/vetterle/skipthought/eval/SICK/'
    SNLI_PATH = '/home/vetterle/InferSent/dataset/SNLI/'
    TORONTO_PATH = '/cluster/project6/mr_corpora/vetterle/toronto/'
    SAVE_PATH = '/cluster/project2/mr/vetterle/thesis'

else:
    # MODEL_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/models/m6/'
    MODEL_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/models/toronto_n5/'

    INFERSENT_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/code/'
    SICK_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/eval/SICK/'
    SNLI_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/dataset/SNLI/'
    TORONTO_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/corpus/'
    SAVE_PATH = './'

# sys.path.append('./')
# sys.path.append(SKIPTHOUGHT_PATH)
# sys.path.append(INFERSENT_PATH)
# from infersent.data import get_nli
# import skipthought
from skipthought.skipthought import Skipthought_para


MODELS = ['skipthought', 'infersent']
TASKS = ['Predict_words', 'Predict_length', 'Predict_dep']

MODEL = MODELS[0]
TASK = TASKS[1]
CBOW = False
UNTRAINED = False

_learning_rate = 0.0001
_batch_size = 64
_epochs = 20
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
            toy = False)
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