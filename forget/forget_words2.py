import tensorflow as tf
import pickle as pkl
import numpy as np
import os
import pandas as pd
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import skipthought.gru_cell as gru_cell
from infersent.data import get_nli

class Forget_words(object):

    def __init__(self, vocab, path, batch_size=128, num_classes=2, uniform_init_scale=0.1, vocabulary_size=20003, embedding_size=620, hidden_size=2400):
        self.vocab = vocab
        self.path = path
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.uniform_init_scale = uniform_init_scale
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        print('\r~~~~~~~ Building graph ~~~~~~~\r')
        self.graph = tf.get_default_graph()
        self.initializer = tf.random_normal_initializer()
        self.uniform_initializer = tf.random_uniform_initializer(
            minval=-self.uniform_init_scale, 
            maxval=self.uniform_init_scale)

        self.word_embeddings = tf.get_variable(
            'embeddings', 
            [self.vocabulary_size, self.embedding_size], 
            tf.float32, 
            initializer = self.initializer)

        self.global_step = tf.Variable(0, 
            name = 'global_step', 
            trainable = False)
        self.sentences = tf.placeholder(tf.int32, 
            [None, None], 
            "sentences")
        self.sentences_lengths = tf.placeholder(tf.int32, 
            [None], 
            "sentences_lengths")
        self.y = tf.placeholder(
            tf.float32, 
            [None, 1], 
            'labels')
        self.words = tf.placeholder(
            tf.int32, 
            [None], 
            'words')

        self.projection_matrix = tf.get_variable('projection', shape=[620, 620])
        self.projected_embeddings = tf.matmul(self.word_embeddings, self.projection_matrix)
        # self.projected_embeddings =  tf.contrib.layers.fully_connected(
        #             self.word_embeddings, 620, activation_fn=None,
        #             scope='projection')

        self.words_embedded = tf.nn.embedding_lookup(self.projected_embeddings, self.words)
        self.sentences_embedded = tf.nn.embedding_lookup(self.projected_embeddings, self.sentences)

        self.encoded_sentences = self.encoder(
            self.sentences_embedded, 
            self.sentences_lengths)

        self.features = tf.concat([self.encoded_sentences, self.words_embedded], axis = 1)
        self.logits = tf.contrib.layers.fully_connected(
                    self.features, 1, activation_fn=None,
                    scope='output_layer')
        self.forget_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=self.y, 
                    logits=self.logits))
        self.eta = tf.train.exponential_decay(
                    0.0001, 
                    self.global_step, 
                    1000, 
                    0.5, 
                    staircase=True)
        self.forget_op = tf.contrib.layers.optimize_loss(
                    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='projection'),
                    # variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'),
                    loss = self.forget_loss, 
                    global_step = self.global_step, 
                    learning_rate = self.eta, 
                    optimizer = 'Adam') 
        forget_loss_sum = tf.summary.scalar('forget_loss', self.forget_loss)

        self.sigmoid = tf.nn.sigmoid(self.logits)
        self.embedding_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embeddings')
        self.projection_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='projection')
        self.encoder_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        self.output_weights = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='output_layer')


        
        self.sess = tf.Session(graph = self.graph)
        self.initialize_uninitialized_vars()
        self.train_loss_writer = tf.summary.FileWriter('./forget/tensorboard/', self.sess.graph)
        self.merged2 = tf.summary.merge([forget_loss_sum])



    def encoder(self, sentences_embedded, sentences_lengths):
        with tf.variable_scope("encoder") as varscope:
            cell = gru_cell.LayerNormGRUCell(
                self.hidden_size,
                w_initializer=self.initializer,
                u_initializer=random_orthonormal_initializer,
                b_initializer=tf.constant_initializer(0.0))
            _, sentences_states = tf.nn.dynamic_rnn(
                cell = cell, 
                inputs = self.sentences_embedded, 
                sequence_length=sentences_lengths, 
                dtype=tf.float32, scope=varscope)   
        return sentences_states

    def initialize_projection_matrix(self):
        self.sess.run(self.projection_matrix.assign(np.eye(620)/10 + np.eye(620)*np.random.randn(620,620)/1000))

    def load_embeddings(self, path):
        word_embeddings = np.load(os.path.join(path,'word_embeddings.npy'))
        self.sess.run(self.word_embeddings.assign(word_embeddings))

    def load_st_encoder(self, path):
        with open(os.path.join(path, 'st_encoder.pkl'), 'rb') as f:
            np_paras = pkl.load(f)
        encoder_vars = [t.name.split(':')[0].split('encoder/')[1] for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')]
        ops = []

        with tf.variable_scope('encoder', reuse=True):
            for a, b in zip(encoder_vars, np_paras):
                ops.append(tf.get_variable(a).assign(b))
        self.sess.run(ops)

    def load_ft(self, path):
        with open(os.path.join(path, 'encoder_forget.pkl'), 'rb') as f:
            np_paras = pkl.load(f)
        encoder_vars = [t.name.split(':')[0].split('encoder/')[1] for t in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')]
        ops = []

        with tf.variable_scope('encoder', reuse=True):
            for a, b in zip(encoder_vars, np_paras):
                ops.append(tf.get_variable(a).assign(b))
        self.sess.run(ops)

    def save_ft_rnn(self, path):

        out = self.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'))
        with open(os.path.join(path, 'encoder_forget.pkl'), 'wb') as f:
            pkl.dump(out, f)

    def save_ft_embeddings(self, path):

        forgetful_embeddings = self.sess.run(self.projected_embeddings)
        np.save(os.path.join(path, 'forgetful_embeddings.npy'), forgetful_embeddings)

        projection_matrix = self.sess.run(self.projection_matrix)
        np.save(os.path.join(path, 'projection_matrix.npy'), projection_matrix)

    def load_output_layer(self, path):

        with open(os.path.join(path, 'output_layer.pkl'), 'rb') as f:
            np_w, np_b = pkl.load(f)[0]
        with tf.variable_scope("output_layer", reuse=True):
            tf_w = tf.get_variable('weights')
            tf_b = tf.get_variable('biases')
        w_op = tf_w.assign(np_w)
        b_op = tf_b.assign(np_b)
        self.sess.run([w_op, b_op])

    def initialize_uninitialized_vars(self):
        from itertools import compress
        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([~(tf.is_variable_initialized(var)) \
                                       for var in global_vars])
        not_initialized_vars = list(compress(global_vars, is_not_initialized))
        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))

    def sent_to_int(self, sentences):

        sentences = [[w for w in sentence.split(' ')] for sentence in sentences]
        sentences_lengths = np.array([len(sentence) for sentence in sentences], dtype=np.int32)

        indices = np.full([len(sentences), np.max(sentences_lengths)], self.vocab['<PAD>'], dtype=np.float32)
        for i, sentence in enumerate(sentences):
            words = [self.vocab[word.lower()] if word in self.vocab else self.vocab['<OOV>'] for word in sentence]
            indices[i, :len(sentence)] = np.array(words)
        return indices, sentences_lengths

    def forget(self, epochs, path, snli_path):

        print('\n~~~~~~~ Loading corpus ~~~~~~~\n')
        train, train_words, dev, dev_words, test, test_words = setup(SNLI_PATH)
        print(len(train))
        train = train + dev + test
        print(len(train))
        train_words = train_words + dev_words + test_words
        print("Trainable variables: {}".format(tf.trainable_variables()))
        mean_prob, min_prob, max_prob= [], [], []

        for epoch in range(epochs):
            print('\n~~~~~~~ Starting training ~~~~~~~\n')
            print('----- Epoch', epoch, '-----')
            print('learning_rate: [{}]'.format(self.sess.run(self.eta)))
            perm = np.random.permutation(len(train))
            train_perm = np.array(train)[perm]
            train_words_perm = np.array(train_words)[perm]
            
            avg_loss = 0
            steps = len(train) // self.batch_size
            for step in range(0, len(train), self.batch_size):

                # print(train_perm[step:(step + model.para.batch_size)][0])
                # print(train_words_perm[step:(step + model.para.batch_size)][0])
                sentences, sentences_lengths = self.sent_to_int(train_perm[step:(step + self.batch_size)])
                words, _ = self.sent_to_int(train_words_perm[step:(step + self.batch_size)])

                labels = np.ones([len(sentences), 1])/2
                feed_dict = {self.sentences: sentences,
                              self.sentences_lengths: sentences_lengths,
                              self.words: np.squeeze(words),
                              self.y: labels}
                # _, batch_loss, current_step = self.sess.run(
                #     [self.forget_op, self.forget_loss, self.global_step], feed_dict=feed_dict)
                _, batch_loss, batch_summary, current_step, a,b,c,d, logits, sigmoid = self.sess.run(
                    [self.forget_op, self.forget_loss, self.merged2, self.global_step, self.output_weights, 
                    self.embedding_weights, self.encoder_weights, self.projection_matrix, self.logits, self.sigmoid], feed_dict=feed_dict)
                min_prob.append(np.min(sigmoid))
                max_prob.append(np.max(sigmoid))
                mean_prob.append(np.mean(sigmoid))
                # print(np.sum([np.sum(i) for i in a]))
                # print(np.sum([np.sum(i) for i in b]))
                # print(np.sum([np.sum(i) for i in c]))
                # print(d[:10,:10])
                # print(np.sum(e))
                # print(labels[:10])
                # print(np.min(d), np.max(d), np.mean(d))
                # print(logits[:10])
                # print(sigmoid[:10])
                # print(labels[:10])

                avg_loss += batch_loss
                print('\rBatch loss at step %d: %0.5f, avg loss: %0.5f, min: %0.2f, max: %0.2f, mean: %0.2f' % (step / self.batch_size, batch_loss, avg_loss/((step+self.batch_size)/ self.batch_size), np.min(sigmoid), np.max(sigmoid), np.mean(sigmoid)), end = '    ')
                self.train_loss_writer.add_summary(batch_summary, current_step)
            # self.save_ft_rnn(path)
            self.save_ft_embeddings(path)
            df = pd.DataFrame({'min': min_prob, 'max': max_prob, 'mean': mean_prob})
            print(len(df))
            df.to_csv(os.path.join(path, 'probabilities.csv'), index=False)

    def get_len(self):

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('./forget/sent_words0False/step_1.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./forget/sent_words0False/'))
            sess.run('bias:0')

def remove_short_sentences(sentences, labels=None, min_len=4):

    lengths = [len(sentence.split()) for sentence in sentences]
    sentences = np.array(sentences)[np.array(lengths) > min_len]
    if labels == None:
        return sentences
    else:
        labels = np.array(labels)[np.array(lengths) > min_len]
        return sentences, labels

def get_relevant(sentences):

    sents, words = [], []
    for s in sentences:
        for w in s.lower().split():
            if w in adj_dic.keys():
                sents.append(s)
                words.append(w)
    return sents, words

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
    train, train_words = get_relevant(remove_short_sentences(train))
    dev, dev_words = get_relevant(remove_short_sentences(dev))
    test, test_words = get_relevant(remove_short_sentences(test))

    return train, train_words, dev, dev_words, test, test_words

def random_orthonormal_initializer(shape, dtype=tf.float32, partition_info=None):  # pylint: disable=unused-argument
    """Variable initializer that produces a random orthonormal matrix."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expecting square shape, got %s" % shape)
    _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
    return u

cluster = True

if cluster:
    MODEL_PATH = '/cluster/project2/mr/vetterle/skipthought/toronto_n5/'
    INFERSENT_PATH = '/home/vetterle/InferSent/code/'
    SICK_PATH = '/home/vetterle/skipthought/eval/SICK/'
    SNLI_PATH = '/home/vetterle/InferSent/dataset/SNLI/'
    TORONTO_PATH = '/cluster/project6/mr_corpora/vetterle/toronto/'
    with open('/home/vetterle/FAIR/InferSent/encoder/adj_dic.pkl', 'rb') as f:
        adj_dic = pkl.load(f)

else:
    MODEL_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/models/toronto_n5/'
    INFERSENT_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/code/'
    SICK_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/eval/SICK/'
    SNLI_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/dataset/SNLI/'
    TORONTO_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/corpus/'
    with open('/Users/Jonas/Documents/Repositories/thesis/adj_dic.pkl', 'rb') as f:
        adj_dic = pkl.load(f)

step = 488268
with open(MODEL_PATH + 'paras.pkl', 'rb') as f:
    paras = pkl.load(f)
with open(MODEL_PATH + 'vocab.pkl', 'rb') as f:
    vocab = pkl.load(f)


MODELS = ['skipthought', 'infersent']
TASKS = ['Predict_words', 'Predict_length', 'Predict_dep']
MODEL = MODELS[0]
TASK = TASKS[0]
CBOW = False
SAVE_PATH = './tasks/saved_models/{}/{}/{}'.format(MODEL, TASK, 'CBOW' if CBOW else 'noCBOW')

tf.reset_default_graph()
model = Forget_words(vocab = vocab, path = MODEL_PATH)
model.initialize_projection_matrix()
model.load_embeddings(path='./')
model.load_st_encoder(path='./')
model.load_output_layer(path = SAVE_PATH)
# model.load_ft(path = SAVE_PATH)

model.forget(epochs = 10, path = SAVE_PATH, snli_path = SNLI_PATH)



# original_vars = tf.global_variables()
# model.sess = tf.Session(graph = model.graph)
# saver = tf.train.Saver()
# saver.restore(model.sess, MODEL_PATH + 'saved_models/step_%d' % step)


# model.randomshit = tf.get_variable(
#             'randomshit', 
#             [model.vocabulary_size, model.para.embedding_size], 
#             tf.float32, 
#             initializer = model.initializer)





# saver = tf.train.Saver([v.name for v in global_vars[:-1]])
# saver.restore(model.sess, MODEL_PATH + 'saved_models/step_%d' % step)



# model.load_model(MODEL_PATH, step)

# new_saver = tf.train.import_meta_graph('/Users/Jonas/Documents/Repositories/thesis/skipthought/models/toronto_n5/saved_models/step_488268.meta')