from skipthought.skipthought import Skipthought_para
from skipthought.skipthought import Skipthought_model
from skipthought import util
import tensorflow as tf
import pickle as pkl
from itertools import compress
import numpy as np
from infersent.data import get_nli
import os

class ForgetThought_model(Skipthought_model):

    def __init__(self, vocab, parameters, path):
        super().__init__(vocab, parameters, path)
        
    def create_ft(self, path, num_classes, step):
        # encoder_vars = [n for n in a if ('precoder' not in n.name) and ('postcoder' not in n.name)]

        self.num_classes = num_classes
        self.load_model(path, step)
        self.word_embeddings = tf.stop_gradient(self.word_embeddings)
        self.y = tf.placeholder(tf.float32, [None, self.num_classes], 'labels')
        self.logits = tf.contrib.layers.fully_connected(
                    self.encoded_sentences, self.num_classes, activation_fn=None,
                    scope='output_layer')
        self.forget_loss = tf.losses.softmax_cross_entropy(
                    onehot_labels=self.y, 
                    logits=self.logits)
        self.eta = tf.train.exponential_decay(
                    self.para.learning_rate, 
                    self.global_step, 
                    self.para.decay_steps, 
                    self.para.decay, 
                    staircase=True)
        self.forget_op = tf.contrib.layers.optimize_loss(
                    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'),
                    loss = self.forget_loss, 
                    global_step = self.global_step, 
                    learning_rate = self.eta, 
                    optimizer = 'Adam') 
        forget_loss_sum = tf.summary.scalar('forget_loss', self.forget_loss)

        self.sess.run(self.global_step.assign(0))
        self.initialize_uninitialized_vars()

        self.train_loss_writer = tf.summary.FileWriter('./forget/tensorboard/', self.sess.graph)
        self.merged2 = tf.summary.merge([forget_loss_sum])


    def load_ft(self):
        return

    def save_ft(self, path):

        out = self.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder'))
        with open(os.path.join(path, 'encoder_forget.pkl'), 'wb') as f:
            pkl.dump(out, f)
        return

    def load_output_layer(self, path):

        with open(os.path.join(path, 'output_layer.pkl'), 'rb') as f:
            np_w, np_b = pkl.load(f)[0]
        with tf.variable_scope("output_layer", reuse=True):
            tf_w = tf.get_variable('weights')
            tf_b = tf.get_variable('biases')
        w_op = tf_w.assign(np_w)
        b_op = tf_b.assign(np_b)
        self.sess.run([w_op, b_op])

# with tf.variable_scope("output_layer", reuse=True):
#     tf_w = tf.get_variable('weights')
# w_var = model.sess.run(tf_w)

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
        train, dev, test = get_nli(snli_path)
        train = np.array(train['s2'])
        dev = np.array(dev['s2'])
        test = np.array(test['s2'])

        for epoch in range(epochs):
            print('\n~~~~~~~ Starting training ~~~~~~~\n')
            print('----- Epoch', epoch, '-----')
            perm = np.random.permutation(len(train))
            train_perm = train[perm]
            
            avg_loss = 0
            steps = len(train) // model.para.batch_size
            for step in range(0, len(train), model.para.batch_size):

                sentences, sentences_lengths = self.sent_to_int(train_perm[step:(step + model.para.batch_size)])
                labels = np.ones([model.para.batch_size, self.num_classes])/self.num_classes
                feed_dict = {self.sentences: sentences,
                              self.sentences_lengths: sentences_lengths,
                              self.y: labels,
                              self.keep_prob_dropout: 1.0}
                # _, batch_loss, current_step = self.sess.run(
                #     [self.forget_op, self.forget_loss, self.global_step], feed_dict=feed_dict)
                _, batch_loss, batch_summary, current_step = self.sess.run(
                    [self.forget_op, self.forget_loss, self.merged2, self.global_step], feed_dict=feed_dict)

                avg_loss += batch_loss/steps
                print('\rBatch loss at step %d: %0.5f' % (step / model.para.batch_size, batch_loss), end = '    ')
                self.train_loss_writer.add_summary(batch_summary, current_step)
                self.save_ft(path)
            # _,_,f1 = self.test_model(X_dev, y_dev)

            # if f1>best_f1:
            #     self.save_model(save_path, 1)   
            # else:
            #     break

    def get_len(self):

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph('./forget/sent_words0False/step_1.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./forget/sent_words0False/'))
            sess.run('bias:0')

cluster = False

if cluster:
    # MODEL_PATH = '/cluster/project2/mr/vetterle/infersent/m6/'
    MODEL_PATH = '/cluster/project2/mr/vetterle/skipthought/toronto_n5/'

    INFERSENT_PATH = '/home/vetterle/InferSent/code/'
    SICK_PATH = '/home/vetterle/skipthought/eval/SICK/'
    SNLI_PATH = '/home/vetterle/InferSent/dataset/SNLI/'
    TORONTO_PATH = '/cluster/project6/mr_corpora/vetterle/toronto/'

else:
    # MODEL_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/models/m6/'
    MODEL_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/models/toronto_n5/'

    INFERSENT_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/code/'
    SICK_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/eval/SICK/'
    SNLI_PATH = '/Users/Jonas/Documents/Repositories/thesis/infersent/dataset/SNLI/'
    TORONTO_PATH = '/Users/Jonas/Documents/Repositories/thesis/skipthought/corpus/'

step = 488268
with open(MODEL_PATH + 'paras.pkl', 'rb') as f:
    paras = pkl.load(f)
with open(MODEL_PATH + 'vocab.pkl', 'rb') as f:
    vocab = pkl.load(f)

MODELS = ['skipthought', 'infersent']
TASKS = ['Predict_words', 'Predict_length', 'Predict_dep']
MODEL = MODELS[0]
TASK = TASKS[1]
CBOW = False
SAVE_PATH = './tasks/saved_models/{}/{}/{}'.format(MODEL, TASK, 'CBOW' if CBOW else 'noCBOW')

tf.reset_default_graph()
model = ForgetThought_model(vocab = vocab, parameters = paras, path = MODEL_PATH)
model.create_ft(path = MODEL_PATH, num_classes = 6, step = step)
model.load_output_layer(path = SAVE_PATH)
model.para.batch_size = 64
model.para.learning_rate = 0.0001
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