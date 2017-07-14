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

# if __name__ == '__main__':

# Parameters for SICK classifier
_learning_rate = 0.01
_epochs = 10000
_batch_size = 64
_train_size = 0.8
_temp_size = 10000
_L2 = 0
_use_expanded_vocab = True # Determines whether to use expanded vocabulary, or the vocabulary that was used for training
step = 450000 # Determines which saved model to use
_keep_prob = 1.0
_hidden_size = 512
_decay = 0.9
_decay_steps = 3000

# path = '../models/toronto_n5/'
path = '/cluster/project2/mr/vetterle/skipthought/toronto_n5/'

with open(path + 'paras.pkl', 'rb') as f:
    paras = pkl.load(f)

with open(path + 'vocab.pkl', 'rb') as f:
    vocab = pkl.load(f)

tf.reset_default_graph()
model = Skipthought_model(vocab = vocab, parameters = paras, path = path)
model.load_model(path, step)

print('Using expanded vocab')
with open(path + 'expanded_vocab.pkl', 'rb') as f:
    vocab = pkl.load(f)


embeddings = np.load(path + 'expanded_embeddings.npy')

def embed(embeddings, data):

# random embeddings
# num = len(vocab)
# embeddings = np.random.randn(num,model.para.embedding_size)
# print('random embeddings created')

    sent_all = data.sentence_A.tolist() + data.sentence_B.tolist()

    sent_lengths, sentences = sick_encode(
        sentences = sent_all[:_temp_size], 
        dictionary = vocab, 
        embeddings = embeddings)

    sentences_encoded = model.encoded_sentences.eval(
            session = model.sess, 
            feed_dict={
                model.sentences_embedded: sentences, 
                model.sentences_lengths: sent_lengths})

    n = np.shape(sentences_encoded)[0]//2
    perm = np.random.permutation(n)

    sent_a = sentences_encoded[:n, :]
    sent_b = sentences_encoded[n:, :]
    feature_1 = sent_a * sent_b 
    feature_2 = np.abs(sent_a - sent_b)
    features = np.concatenate(
        (feature_1, feature_2), axis=1)

    return features

data = pd.read_csv('../eval/SICK/SICK_train.txt', sep='\t', index_col=0)
train_data = embed(embeddings, data)
score = data.relatedness_score.tolist()
train_labels = encode_labels(score[:_temp_size])

data = pd.read_csv('../eval/SICK/SICK_trial.txt', sep='\t', index_col=0)
trial_data = embed(embeddings, data)
score = data.relatedness_score.tolist()
trial_labels = encode_labels(score[:_temp_size])

train_data = np.concatenate((train_data, trial_data), axis=0)
train_labels = np.concatenate((train_labels, trial_labels), axis=0)

data = pd.read_csv('../eval/SICK/SICK_test_annotated.txt', sep='\t', index_col=0)
test_data = embed(embeddings, data)
test_labels = data.relatedness_score.tolist()
test_labels_encoded = encode_labels(test_labels)

n_train = len(train_data)
n_test = len(test_data)

print('\nFound %d training examples and %d test examples\n' %(n_train, n_test))

tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    # initializer = tf.random_normal_initializer()
    initializer = tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)

    sick_scores = tf.placeholder(
        tf.int32, 
        [None, None], 
        'sick_scores')
    sick_features = tf.placeholder(
        tf.float32, 
        [None, None], 
        'features')

    b = tf.get_variable('sick_bias', 
        [5], 
        tf.float32, 
        initializer = initializer)

    W = tf.get_variable('sick_weight', 
        [4800, 5], 
        tf.float32, 
        initializer = initializer)

    r = tf.reshape(tf.range(1, 6, 1, 
        dtype = tf.float32), 
        [-1, 1])
    global_step = tf.Variable(0, 
        name = 'global_step', 
        trainable = False)

    logits = tf.matmul(sick_features, W) + b

    all_vars = tf.trainable_variables() 
    l2_reg = tf.add_n([ tf.nn.l2_loss(v) for v in all_vars if 'bias' not in v.name ]) * _L2
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels = sick_scores, 
            logits=logits)) + l2_reg
    eta = tf.train.exponential_decay(
        _learning_rate, 
        global_step, 
        _decay_steps, 
        _decay, 
        staircase=True)
    opt_op = tf.contrib.layers.optimize_loss(
        loss = loss, 
        learning_rate = eta, 
        optimizer = 'SGD', 
        global_step = global_step) 
    prediction = tf.matmul(tf.nn.softmax(logits), r)

    with tf.Session(graph=graph) as sess:
        tf.global_variables_initializer().run(session = sess)
        for epoch in range(_epochs):
            perm = np.random.permutation(n_train)
            train_data_perm = train_data[perm]
            train_labels_perm = train_labels[perm]
            avg_loss = 0
            steps = n_train//_batch_size
            for step in range(steps):
                begin = step * _batch_size
                end = (step + 1) * _batch_size
                batch_features = train_data_perm[begin : end]
                batch_score_encoded = train_labels_perm[begin : end]
                train_dict = {sick_scores: batch_score_encoded,
                              sick_features: batch_features}
                _, batch_loss, batch_prediction = sess.run(
                    [opt_op, loss, prediction], 
                    feed_dict=train_dict)
                avg_loss += batch_loss/steps
                print('\rBatch loss: %0.2f' % batch_loss, end = '    ')
                # print(batch_loss)

            # train_dict = {sick_scores: train_labels_perm,
            #                   sick_features: train_data_perm}
            # _, batch_loss, batch_prediction = sess.run(
            #         [opt_op, loss, prediction], 
            #         feed_dict=train_dict)


            # avg_loss += batch_loss
            # print('\rBatch loss: %0.2f' % batch_loss, end = '    ')

            if epoch % 1==0:
                test_dict = {sick_scores: test_labels_encoded,
                        sick_features: test_data}
                test_loss, test_prediction, lr = sess.run(
                    [loss, prediction, eta], 
                    feed_dict=test_dict)
                pr = pearsonr(test_prediction[:,0], test_labels)[0]
                sr = spearmanr(test_prediction[:,0], test_labels)[0]
                se = mse(test_prediction[:,0], test_labels)
                print('\nEpoch %d: Train loss: %0.2f, Dev loss: %0.2f, Dev pearson: %0.2f, Dev spearman: %0.2f, Dev MSE: %0.2f, lr: %0.4f\n' 
                    % (epoch, avg_loss, test_loss, pr, sr, se, lr))


                # i = np.random.randint(len(model.sentences_embedded))
                # print(model.print_sentence(model.sentences_embedded[i], model.enc_lengths[i]))
