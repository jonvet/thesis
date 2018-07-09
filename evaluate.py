import tensorflow as tf
import os
import tasks.predict_dep as predict_dep
import tasks.predict_length as predict_length
import tasks.predict_words as predict_words
import tasks.predict_sent as predict_sent
import encoder

cluster = False

if cluster:
    MODEL_PATH = '/cluster/project2/mr/vetterle/skipthought/toronto_n5/'
    INFERSENT_PATH = '/home/vetterle/InferSent/code/'
    SICK_PATH = '/home/vetterle/skipthought/eval/SICK/'
    SNLI_PATH = '/home/vetterle/InferSent/dataset/SNLI/'
    TORONTO_PATH = '/cluster/project6/mr_corpora/vetterle/toronto/'
    SAVE_PATH = '/cluster/project2/mr/vetterle/thesis'

else:
    MODEL_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/models/toronto_n5/'
    INFERSENT_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/code/'
    SICK_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/eval/SICK/'
    SNLI_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/dataset/SNLI/'
    TORONTO_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/corpus/'
    SAVE_PATH = './'

MODELS = ['skipthought', 'infersent']
TASKS = ['Predict_words', 'Predict_length', 'Predict_dep', 'Predict_sent']

MODEL = MODELS[0]
TASK = TASKS[3]
CBOW = False
UNTRAINED = False

_learning_rate = 0.0001
_batch_size = 64
_epochs = 20
_dropout = 0.9
        
MODE = 'train'

if __name__ == '__main__':

    if TASK=='Predict_length':

        CBOW = False

        SAVE_PATH = './tasks/saved_models/{}/{}/{}'.format(MODEL, TASK, 'CBOW' if CBOW else 'noCBOW')
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        tf.reset_default_graph()
        enc = encoder.Encoder(
            model_name = MODEL, 
            model_path = MODEL_PATH, 
            cbow = CBOW,
            snli_path = SNLI_PATH)
        train, dev, test = predict_length.setup(
            snli_path = SNLI_PATH,
            toy = False)
        task = predict_length.Predict_length(
            encoder = enc,
            learning_rate = _learning_rate,
            epochs=_epochs)

        # if MODE == 'train':
        #     task.train_model(train, dev, y_train = None, y_dev = None, save_path = SAVE_PATH)
        # elif MODE == 'test':
        #     task.load_output_layer(path = SAVE_PATH)
        #     _,_,_ = task.test_model(test, None)
        # elif MODE == 'test_forget':
        #     task.load_output_layer(path = SAVE_PATH)
        #     task.load_ft(path = SAVE_PATH) 
        #     _,_,_ = task.test_model(test, None)

        task.sess = tf.Session(graph = task.graph)
        tf.global_variables_initializer().run(session = task.sess)
        _,_,_ = task.test_model(test, None)

        # task.load_output_layer(path = SAVE_PATH)
        # _,_,_ = task.test_model(test, None)

        # task.load_ft(path = SAVE_PATH)
        # _,_,_ = task.test_model(test, None)

    elif TASK=='Predict_sent':

        CBOW = False

        SAVE_PATH = './tasks/saved_models/{}/{}/{}'.format(MODEL, TASK, 'CBOW' if CBOW else 'noCBOW')
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        tf.reset_default_graph()
        enc = encoder.Encoder(
            model_name = MODEL, 
            model_path = MODEL_PATH, 
            cbow = CBOW,
            snli_path = SNLI_PATH)
        train, dev, test = predict_sent.setup(
            snli_path = SNLI_PATH,
            toy = False)
        task = predict_sent.Predict_sent(
            encoder = enc,
            learning_rate = _learning_rate,
            epochs=_epochs)

        if MODE == 'train':
            task.train_model(train, dev, y_train = None, y_dev = None, save_path = SAVE_PATH)
        elif MODE == 'test':
            task.load_output_layer(path = SAVE_PATH)
            _,_,_ = task.test_model(test, None)
        elif MODE == 'test_forget':
            task.load_output_layer(path = SAVE_PATH)
            task.load_ft(path = SAVE_PATH) 
            _,_,_ = task.test_model(test, None)



    elif TASK=='Predict_words':

        CBOW = False

        SAVE_PATH = './tasks/saved_models/{}/{}/{}'.format(MODEL, TASK, 'CBOW' if CBOW else 'noCBOW')
        if not os.path.exists(SAVE_PATH):
            os.makedirs(SAVE_PATH)

        tf.reset_default_graph()
        enc = encoder.Encoder(
            model_name = MODEL, 
            model_path = MODEL_PATH, 
            cbow = CBOW,
            snli_path = SNLI_PATH)
        train, dev, test, pos, neg = predict_words.setup(
            snli_path = SNLI_PATH,
            toy = False)
        task = predict_words.Predict_words(
            encoder = enc,
            learning_rate = _learning_rate,
            epochs=_epochs)

        if MODE == 'train':
            task.train_model(X_train=train, X_dev=dev, y_train=[pos[0], neg[0]], y_dev=[pos[1], neg[1]], save_path = SAVE_PATH)
        elif MODE == 'test':
            task.load_output_layer(path = SAVE_PATH)
            _,_,_ = task.test_model(test, None)
        elif MODE == 'test_forget':
            task.load_output_layer(path = SAVE_PATH)
            task.load_ft(path = SAVE_PATH) 
            _,_,_ = task.test_model(test, None)


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
