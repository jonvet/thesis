import tensorflow as tf
import numpy as np
from util import sick_encode
from skipthought import Skipthought_para
from skipthought import Skipthought_model
import pandas as pd
import pickle as pkl
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error as mse

def encode_labels(labels, nclass=5):
    """
    https://github.com/ryankiros/skip-thoughts/
    """
    Y = np.zeros((len(labels), nclass)).astype('float32')
    for j, y in enumerate(labels):
        for i in range(nclass):
            if i+1 == np.floor(y) + 1:
                Y[j,i] = y - np.floor(y)
            if i+1 == np.floor(y):
                Y[j,i] = np.floor(y) - y + 1
    return Y    

def prepare_batches(lengths, sentences, b_size):

    num_batches = len(lengths) // b_size

    b_lengths = []
    b_sentences = []
    for i in range(num_batches+1):
        b_lengths.append(lengths[i*b_size:(i+1)*b_size])
        b_sentences.append(sentences[i*b_size:(i+1)*b_size,:])
    return b_lengths, b_sentences

if __name__ == '__main__':

    # Parameters for SICK classifier
    _hidden_size = 700
    _learning_rate = 0.05
    _epochs = 500
    _batch_size = 25
    _train_size = 0.8
    _temp_size = 10000
    _use_expanded_vocab = True # Determines whether to use expanded vocabulary, or the vocabulary that was used for training
    epoch = 1 # Determines which saved model to use


    path = './models/toronto_n2/'

    data = pd.read_csv('./eval/SICK/SICK_train.txt', sep='\t', index_col=0)
    sent_all = data.sentence_A.tolist() + data.sentence_B.tolist()

    with open(path + 'paras.pkl', 'rb') as f:
        paras = pkl.load(f)

    with open(path + 'vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)

    tf.reset_default_graph()
    model = Skipthought_model(vocab = vocab, parameters = paras, path = path)
    model.load_model(path, epoch)
    
    print('Using skipthought model to encode SICK sentences')
    if _use_expanded_vocab:
        print('Using expanded vocab')
        with open(path + 'expanded_vocab.pkl', 'rb') as f:
            vocab = pkl.load(f)

        # random embeddings
        num = len(vocab)
        embeddings = np.random.randn(num,model.para.hidden_size)
        # embeddings = np.load(path + 'expanded_embeddings.npy')

        sent_lengths, sentences = sick_encode(
            sentences = sent_all[:_temp_size], 
            dictionary = vocab, 
            embeddings = embeddings)

        b_lengths, b_sentences = prepare_batches(
            lengths=sent_lengths, 
            sentences=sentences, 
            b_size=32)

        sentences_encoded = []

        print('\nComputing skipthought sentence representations...')
        for batch in range(len(b_lengths)):
            print('\rBatch %d/%d' %(batch, len(b_lengths)), end='   ')
            b_sentences_encoded = model.encoded_sentences.eval(
                session = model.sess, 
                feed_dict={
                    model.sentences_embedded: b_sentences[batch], 
                    model.sentences_lengths: b_lengths[batch]})
            sentences_encoded.append(b_sentences_encoded)
        sentences_encoded = np.concatenate([np.array(i) for i in sentences_encoded])
    else:
        sent_lengths, sentences = sick_encode(
            sentences = sent_all[:_temp_size], 
            dictionary = vocab)
        model.enc_lengths, model.enc_data = sent_lengths, sentences
        sentences_encoded = model.encode(sentences, sent_lengths)

    print(np.shape(sentences_encoded))    
    n = np.shape(sentences_encoded)[0]//2
    
    print('\nCreating SICK features')
    sent_a = sentences_encoded[:n, :]
    sent_b = sentences_encoded[n:, :]
    feature_1 = sent_a * sent_b 
    feature_2 = np.abs(sent_a - sent_b)

    score = data.relatedness_score.tolist()
    score_encoded = encode_labels(score[:_temp_size])

    perm = np.random.permutation(n)
    n_train = int(np.floor(0.8*n))
    train_feature_1 = feature_1[perm][:n_train]
    train_feature_2 = feature_2[perm][:n_train]
    train_score_encoded = score_encoded[perm][:n_train]
    dev_feature_1 = feature_1[perm][n_train:]
    dev_feature_2 = feature_2[perm][n_train:]
    dev_score_encoded = score_encoded[perm][n_train:]
    dev_score = np.array(score)[perm][n_train:]

tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    initializer = tf.random_normal_initializer()

    sick_scores = tf.placeholder(
        tf.int32, 
        [None, None], 
        'sick_scores')
    sick_feature_1 = tf.placeholder(
        tf.float32, 
        [None, None], 
        'sick_feature_1')
    sick_feature_2 = tf.placeholder(
        tf.float32, 
        [None, None], 
        'sick_feature_2')  

    b_1 = tf.get_variable('sick_bias_1', 
        [_hidden_size], 
        tf.float32, 
        initializer = initializer)
    b_2 = tf.get_variable('sick_bias_2', 
        [5], 
        tf.float32, 
        initializer = initializer)
    W_1 = tf.get_variable('sick_weight_1', 
        [model.para.hidden_size, _hidden_size], 
        tf.float32, 
        initializer = initializer)
    W_2 = tf.get_variable('sick_weight_2', 
        [model.para.hidden_size, _hidden_size], 
        tf.float32, 
        initializer = initializer)
    W_3 = tf.get_variable('sick_weight_3', 
        [_hidden_size, 5], 
        tf.float32, 
        initializer = initializer)
    r = tf.reshape(tf.range(1, 6, 1, 
        dtype = tf.float32), 
        [-1, 1])
    global_step = tf.Variable(0, 
        name = 'global_step', 
        trainable = False)

    hidden = tf.sigmoid(tf.matmul(sick_feature_1, W_1) + 
        tf.matmul(sick_feature_2, W_2) + b_1)
    logits = tf.matmul(hidden, W_3) + b_2
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels = sick_scores, 
            logits=logits))
    opt_op = tf.contrib.layers.optimize_loss(
        loss = loss, 
        learning_rate = _learning_rate, 
        optimizer = 'Adagrad', 
        global_step = global_step) 
    prediction = tf.matmul(tf.nn.softmax(logits), r)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run(session = sess)
        for epoch in range(_epochs):
            perm = np.random.permutation(n_train)
            feature_1_perm = train_feature_1[perm]
            feature_2_perm = train_feature_2[perm]
            score_encoded_perm = train_score_encoded[perm]
            avg_loss = 0
            steps = n_train//_batch_size
            for step in range(steps):
                begin = step * _batch_size
                end = (step + 1) * _batch_size
                batch_feature_1 = feature_1_perm[begin : end]
                batch_feature_2 = feature_2_perm[begin : end]
                batch_score_encoded = score_encoded_perm[begin : end]
                train_dict = {sick_scores: batch_score_encoded,
                              sick_feature_1: batch_feature_1,
                              sick_feature_2: batch_feature_2}
                _, batch_loss, batch_prediction = sess.run(
                    [opt_op, loss, prediction], 
                    feed_dict=train_dict)
                avg_loss += batch_loss/steps
                print('\rBatch loss: %0.2f' % batch_loss, end = '    ')

            if epoch % 10==0:
                dev_dict = {sick_scores: dev_score_encoded,
                        sick_feature_1: dev_feature_1,
                        sick_feature_2: dev_feature_2}
                _, dev_loss, dev_prediction = sess.run(
                    [opt_op, loss, prediction], 
                    feed_dict=dev_dict)
                pr = pearsonr(dev_prediction[:,0], dev_score)[0]
                sr = spearmanr(dev_prediction[:,0], dev_score)[0]
                se = mse(dev_prediction[:,0], dev_score)
                print('\nEpoch %d: \
                    Train loss: %0.2f, \
                    Dev loss: %0.2f, \
                    Dev pearson: %0.2f, \
                    Dev spearman: %0.2f, \
                    Dev MSE: %0.2f\n' 
                    % (epoch, avg_loss, dev_loss, pr, sr, se))

                    # i = np.random.randint(len(model.sentences_embedded))
                    # print(model.print_sentence(model.sentences_embedded[i], model.enc_lengths[i]))