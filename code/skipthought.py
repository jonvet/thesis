import tensorflow as tf
import numpy as np
import util
from util import word_vocab
from util import sent_to_int
from util import finalise_vocab
from tensorflow.contrib.tensorboard.plugins import projector
import time
from sys import stdout
import os
import glob
import operator
import csv
import pickle as pkl
from collections import defaultdict
import gru_cell
from gru_cell import NoNormGRUCell
from gru_cell import b_NoNormGRUCell
from gru_cell import b_NoNormGRUCell2

class Skipthought_para(object):

    def __init__(self, embedding_size, hidden_size, hidden_layers, batch_size, keep_prob_dropout, learning_rate, 
            bidirectional, decay_steps, decay, predict_step, max_sent_len, uniform_init_scale, clip_gradient_norm, save_every):
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.keep_prob_dropout = keep_prob_dropout
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.decay_steps = decay_steps
        self.decay = decay
        self.predict_step = predict_step
        self.max_sent_len = max_sent_len
        self.uniform_init_scale = uniform_init_scale
        self.clip_gradient_norm = clip_gradient_norm
        self.save_every = save_every

class Skipthought_model(object):

    def __init__(self, vocab, parameters, path):
        self.para = parameters
        self.vocab = vocab
        self.reverse_vocab = dict(zip(
            self.vocab.values(), 
            self.vocab.keys()))
        self.sorted_vocab = sorted(self.vocab.items(), 
            key=operator.itemgetter(1))
        self.vocabulary_size = len(self.sorted_vocab)
        self.path = path
        self.epoch = 0
        
        print('\r~~~~~~~ Building graph ~~~~~~~\r')
        self.graph = tf.get_default_graph()
        self.initializer = tf.random_normal_initializer()
        self.uniform_initializer = tf.random_uniform_initializer(
            minval=-self.para.uniform_init_scale, 
            maxval=self.para.uniform_init_scale)

        self.word_embeddings = tf.get_variable(
            'embeddings', 
            [self.vocabulary_size, self.para.embedding_size], 
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

        self.post_inputs = tf.placeholder(tf.int32, 
            [None, None], 
            "post_inputs")
        self.post_labels = tf.placeholder(tf.int32, 
            [None, None], 
            "post_labels")
        self.post_sentences_masks = tf.placeholder(tf.int32, 
            [None, None], 
            "post_sentences_masks")

        self.pre_inputs = tf.placeholder(tf.int32, 
            [None, None], 
            "pre_inputs")
        self.pre_labels = tf.placeholder(tf.int32, 
            [None, None], 
            "pre_labels")
        self.pre_sentences_masks = tf.placeholder(tf.int32, 
            [None, None], 
            "pre_sentences_masks")

        self.sentences_embedded = tf.nn.embedding_lookup(self.word_embeddings, self.sentences)
        self.post_inputs_embedded = tf.nn.embedding_lookup(self.word_embeddings, self.post_inputs)
        self.pre_inputs_embedded = tf.nn.embedding_lookup(self.word_embeddings, self.pre_inputs)

        self.encoded_sentences = self.encoder(
            self.sentences_embedded, 
            self.sentences_lengths, 
            self.para.bidirectional)

        self.pre_prob, pre_loss, self.pre_output = self.decoder(
            name = "precoder", 
            decoder_input = self.pre_inputs_embedded,
            targets = self.pre_labels, 
            mask = self.pre_sentences_masks,
            initial_state = self.encoded_sentences, 
            reuse_logits = False)

        self.post_prob, post_loss, self.post_output = self.decoder(
            name = "postcoder", 
            decoder_input = self.post_inputs_embedded,
            targets = self.post_labels, 
            mask = self.post_sentences_masks,
            initial_state = self.encoded_sentences, 
            reuse_logits = True)

        self.loss = pre_loss + post_loss
        self.eta = tf.train.exponential_decay(
            self.para.learning_rate, 
            self.global_step, 
            self.para.decay_steps, 
            self.para.decay, 
            staircase=True)
        self.opt_op = tf.contrib.layers.optimize_loss(
            loss = self.loss, 
            global_step = self.global_step, 
            learning_rate = self.eta, 
            optimizer = 'Adam', 
            clip_gradients=self.para.clip_gradient_norm, 
            summaries = ['loss']) 

        self.sess = tf.Session(graph = self.graph)

    def encoder(self, sentences_embedded, sentences_lengths, bidirectional = False):
        with tf.variable_scope("encoder") as varscope:
            if bidirectional:
                cell = tf.contrib.rnn.GRUCell(self.para.hidden_size//2)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = self.para.keep_prob_dropout)
                cell = tf.contrib.rnn.MultiRNNCell([cell]*self.para.hidden_layers, state_is_tuple=True)
                print('Using bidirectional RNN')
                sentences_outputs, sentences_states = tf.nn.bidirectional_dynamic_rnn(
                    cell, cell, 
                    inputs = self.sentences_embedded, 
                    sequence_length=sentences_lengths, 
                    dtype=tf.float32, 
                    scope = varscope)
                states_fw, states_bw = sentences_states
                sentences_states_h = tf.concat([states_fw[-1],states_bw[-1]], axis = 1)
                # sentences_states_h = tf.contrib.layers.linear(sentences_states_h, self.para.hidden_size)
            else:
                # cell = tf.contrib.rnn.GRUCell(self.para.hidden_size)
                cell = gru_cell.LayerNormGRUCell(
                    self.para.hidden_size,
                    w_initializer=self.initializer,
                    u_initializer=random_orthonormal_initializer,
                    b_initializer=tf.constant_initializer(0.0))
                # cell = NoNormGRUCell(
                #     self.para.hidden_size,
                #     w_initializer=self.uniform_initializer,
                #     u_initializer=random_orthonormal_initializer,
                #     b_initializer=tf.constant_initializer(0.0))
                # cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = self.para.keep_prob_dropout)
                # cell = tf.contrib.rnn.MultiRNNCell([cell]*self.para.hidden_layers, state_is_tuple=True)
                print('Using one-directional RNN')
                _, sentences_states = tf.nn.dynamic_rnn(
                    cell = cell, 
                    inputs = self.sentences_embedded, 
                    sequence_length=sentences_lengths, 
                    dtype=tf.float32, scope = varscope)   
                # sentences_states_h = sentences_states[-1]
        return sentences_states

    # def output_fn(outputs):
    #     return tf.contrib.layers.linear(outputs, self.vocab_size, scope=scope)

    def decoder(self, name, decoder_input, targets, mask, initial_state, reuse_logits):

        # cell = NoNormGRUCell(
        #     self.para.hidden_size,
        #     w_initializer=self.uniform_initializer,
        #     u_initializer=random_orthonormal_initializer,
        #     b_initializer=tf.constant_initializer(0.0))

        cell = gru_cell.LayerNormGRUCell(
            self.para.hidden_size,
            w_initializer=self.initializer,
            u_initializer=random_orthonormal_initializer,
            b_initializer=tf.constant_initializer(0.0))

        with tf.variable_scope(name) as scope:
      
            length = tf.reduce_sum(mask, 1, name="length")
            decoder_output, _ = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=decoder_input,
                sequence_length=length,
                initial_state=initial_state,
                scope=scope)

        decoder_output_reshaped = tf.reshape(decoder_output, [-1, self.para.hidden_size])
        targets = tf.reshape(targets, [-1])
        weights = tf.to_float(tf.reshape(mask, [-1]))

        with tf.variable_scope("logits", reuse=reuse_logits) as scope:
            logits = tf.contrib.layers.fully_connected(
                inputs=decoder_output_reshaped,
                num_outputs=self.vocabulary_size,
                activation_fn=None,
                weights_initializer=self.uniform_initializer,
                scope=scope)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=targets, 
            logits=logits)
        loss = tf.reduce_mean(loss * weights)

        # predict = tf.arg_max(logits, 1)
        probabilities = tf.nn.softmax(logits)

        return probabilities, loss, decoder_output

    def print_sentence(self, sentence, length):
        s = ''
        for i in range(length):
            word = sentence[i]
            s = s+self.reverse_vocab[word]+' '
        return s

    def save_model(self, path, step):
        if not os.path.exists(path):
            os.mkdir(path)
        self.saver.save(sess = self.sess, save_path = path + '/step_%d' % step, write_state = False)

    def load_model(self, path, step):
        self.sess = tf.Session(graph = self.graph)
        saver = tf.train.Saver()
        saver.restore(self.sess, path + '/saved_models/step_%d' % step)
        print('Model restored')

    def evaluate(self, data, mode = 'max', index = None):
        enc_lengths, enc_data, post_data, post_lab, pre_data, pre_lab, post_masks, pre_masks = data[2:]
        i = index if index != None else np.random.randint(len(enc_data))
        print('\nOriginal sequence:')
        print(self.print_sentence(pre_data[i, 1:], np.sum(pre_masks, axis=1)[i]-1))
        print(self.print_sentence(enc_data[i], enc_lengths[i]))
        print(self.print_sentence(post_data[i, 1:], np.sum(post_masks, axis=1)[i]-1))
        test_enc_lengths = np.expand_dims(enc_lengths[i], 0)
        test_enc_inputs = np.expand_dims(enc_data[i], 0)
        test_encoder_state = self.sess.run(
            self.encoded_sentences, 
            feed_dict={self.sentences: test_enc_inputs, self.sentences_lengths: test_enc_lengths})

        decoder_state = test_encoder_state
        decoder_input = np.array([0])
        done = False
        sentence = []
        while done == False:
            l = len(sentence) + 1
            test_dict = {
                self.pre_sentences_masks: np.array([[l]]),
                self.encoded_sentences: decoder_state,
                self.pre_inputs: np.array([decoder_input])}
            decoder_input, decoder_state = self.sess.run(
                [self.pre_prob,  self.pre_output[0]], 
                feed_dict=test_dict)
            if mode == 'max':
                decoder_input = np.argmax(decoder_input)
            else:
                # words = np.arange(1, self.vocabulary_size+1, 1)
                # print(decoder_input.shape)
                # print(self.vocabulary_size+1)
                decoder_input = np.random.choice(self.vocabulary_size, p = decoder_input[0])
            sentence.append(decoder_input)
            decoder_input = np.expand_dims(decoder_input, 0)
            done = True if (decoder_input == 2 or l > self.para.max_sent_len) else False

        print('\nPredicted previous and following sentences around original (middle) sentence:')
        print(self.print_sentence(sentence, len(sentence)))
        print(self.print_sentence(enc_data[i], enc_lengths[i]))

        decoder_state = test_encoder_state
        decoder_input = np.array([0])
        done = False
        sentence = []
        while done == False:
            l = len(sentence) + 1
            test_dict = {
                self.post_sentences_masks: np.array([[l]]),
                self.encoded_sentences: decoder_state,
                self.post_inputs: np.array([decoder_input])}
            decoder_input, decoder_state = self.sess.run(
                [self.post_prob,  self.post_output[0]], 
                feed_dict=test_dict)
            if mode == 'max':
                decoder_input = np.argmax(decoder_input)
            else:
                # words = np.arange(1, self.vocabulary_size+1, 1)
                decoder_input = np.random.choice(self.vocabulary_size, p = decoder_input[0])
            sentence.append(decoder_input)
            decoder_input = np.expand_dims(decoder_input, 0)
            done = True if (decoder_input == 2 or l > self.para.max_sent_len) else False
        print(self.print_sentence(sentence, len(sentence)))

    def encode(self, sentences, lengths):
        encode_dict = {self.sentences: sentences,
                       self.sentences_lengths: lengths}
        encoded_sentences = self.sess.run(self.encoded_sentences, feed_dict=encode_dict)
        return np.array(encoded_sentences)

    def initialise(self):
        # Save metadata for visualisation of embedding matrix
        # meta_data = sorted(self.dictionary, key=self.dictionary.get)
        # print(len(meta_data))
        # with open('meta_data.tsv', 'w') as f:
        #     tsv_writer = csv.writer(f, dialect='excel')
        #     tsv_writer.writerow(str(i.encode('utf-8')) +'\n' for i in meta_data)

        # Print summary statistics
        
        # self.a= tf.contrib.graph_editor.get_tensors(self.graph)
        self.train_loss_writer = tf.summary.FileWriter(self.path + 'tensorboard/', self.sess.graph)
        self.merged = tf.summary.merge_all()
        self.saver = tf.train.Saver()
        self.total_loss = 0
        self.start_time = time.time()

    def train(self, data):
        batch_time = time.time()
        enc_lengths, enc_data, post_data, post_lab, pre_data, pre_lab, post_masks, pre_masks = data[2:]
        try:
            self.corpus_length = len(enc_data)
            perm = np.random.permutation(self.corpus_length)
            enc_lengths_perm = enc_lengths[perm]
            enc_data_perm = enc_data[perm]
            post_masks_perm = np.array(post_masks)[perm]
            post_inputs_perm = np.array(post_data)[perm]
            post_labels_perm = np.array(post_lab)[perm]
            pre_masks_perm = np.array(pre_masks)[perm]
            pre_inputs_perm = np.array(pre_data)[perm]
            pre_labels_perm = np.array(pre_lab)[perm]
            
            n_steps = self.corpus_length // self.para.batch_size
            for step in range(n_steps):
                begin = step * self.para.batch_size
                end = (step + 1) * self.para.batch_size
                batch_enc_lengths = enc_lengths_perm[begin : end]
                batch_enc_inputs = enc_data_perm[begin : end]
                batch_post_masks = post_masks_perm[begin : end]
                batch_post_inputs = post_inputs_perm[begin:end]
                batch_post_labels = post_labels_perm[begin:end]
                batch_pre_masks = pre_masks_perm[begin : end]
                batch_pre_inputs = pre_inputs_perm[begin:end]
                batch_pre_labels = pre_labels_perm[begin:end]

                train_dict = {
                    self.sentences_lengths: batch_enc_lengths,
                    self.sentences: batch_enc_inputs, 
                    self.post_sentences_masks: batch_post_masks,
                    self.post_inputs: batch_post_inputs,
                    self.post_labels: batch_post_labels,
                    self.pre_sentences_masks: batch_pre_masks,
                    self.pre_inputs: batch_pre_inputs,
                    self.pre_labels: batch_pre_labels}
                _, loss_val, batch_summary, current_step = self.sess.run(
                    [self.opt_op, self.loss, self.merged, self.global_step], 
                    feed_dict=train_dict)

                print('\rStep %d loss: %0.5f' % (current_step, loss_val), end='   ')

                self.train_loss_writer.add_summary(batch_summary, current_step)
                self.total_loss += loss_val
                if current_step % self.para.predict_step == 0:
                    print("\nAverage loss at epoch %d step %d:" % (self.epoch, current_step), self.total_loss/self.para.predict_step)
                    print('Learning rate: %0.6f' % self.eta.eval(session = self.sess))
                    self.total_loss = 0
                    end_time = time.time()
                    print('Time for %d steps: %0.2f seconds' % (self.para.predict_step, end_time - batch_time))
                    batch_time = time.time()
                    secs = time.time() - self.start_time
                    hours = secs//3600
                    minutes = secs / 60 - hours * 60
                    print('Time elapsed: %d:%02d hours' % (hours, minutes))
                    print('\n~~~~~~~~~~ Decoding by sampling ~~~~~~~~~~')
                    self.evaluate(data, mode='sample')
                    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
                if current_step % self.para.save_every == 0:
                    self.save_model(self.path + '/saved_models/', current_step)

        except KeyboardInterrupt:
            save = input('save?')
            if 'y' in save:
                self.save_model(self.path + '/saved_models/', self.global_step.eval(session = self.sess))

def random_orthonormal_initializer(shape, dtype=tf.float32, partition_info=None):  # pylint: disable=unused-argument
    """Variable initializer that produces a random orthonormal matrix."""
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("Expecting square shape, got %s" % shape)
    _, u, _ = tf.svd(tf.random_normal(shape, dtype=dtype), full_matrices=True)
    return u

def preprocess(corpus_name, model_path, corpus_path, final_path, vocab_size, max_sent_len):

    '''
    Needs to be run only once.
    This routine will create a vocabulary and skipthought data.
    input_path should contain .txt files with tokenised sentences, one per line.
    '''
    # parts = glob.glob(corpus_path + '*.txt')
    parts = glob.glob(corpus_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(final_path):
        os.makedirs(final_path)
    
    if os.path.isfile(model_path + 'vocab.pkl'):
        with open(model_path + 'vocab.pkl', 'rb') as f:
            vocab = pkl.load(f)
        print('Vocab loaded')
    else:
        print('\nCreating vocab')
        print('\n%d files to be processed:' % len(parts), parts)
        vocab = defaultdict(int)
        for part in parts:
            print('\nProcessing file:', part)
            vocab = word_vocab(part, vocab_name=part, vocab=vocab)
        vocab = finalise_vocab(vocab, vocab_size)
        with open(model_path + 'vocab.pkl', 'wb') as f:
            pkl.dump(vocab, f)
        print('\nVocab created')

    print('\nCreating training data')
    print('\n%d files to be processed:' % len(parts), parts)
    i = 0
    for part in parts:
        print('\nProcessing file:', part)
        data = get_training_data(part, vocab, corpus_name, max_sent_len)
        with open(final_path + 'data_%d.pkl' %i, 'wb') as f:
            pkl.dump(data, f)
        i+=1
    print('\nTraining data created')

def get_training_data(path, vocab, corpus_name, max_sent_len):

    '''
    Create datasets for encoder and decoders
    '''

    sent_lengths, max_sent_len, enc_data, dec_data, dec_lab, dec_masks = sent_to_int(path, dictionary=vocab, max_sent_len=max_sent_len, decoder=True)
    enc_lengths = sent_lengths[1:-1] 
    enc_data = enc_data[1:-1]

    post_data = dec_data[2:]
    post_lab = dec_lab[2:]
    post_masks = dec_masks[2:]

    pre_data = dec_data[:-2]
    pre_lab = dec_lab[:-2]
    pre_masks = dec_masks[:-2]


    return [corpus_name, max_sent_len, enc_lengths, enc_data, post_data, post_lab, pre_data, pre_lab, post_masks, pre_masks]

def make_paras(path):
    if not os.path.exists(path):
        os.makedirs(path)
    paras = Skipthought_para(embedding_size = 620, 
        hidden_size = 2400, 
        hidden_layers = 1, 
        batch_size = 128, 
        keep_prob_dropout = 1.0, 
        learning_rate = 0.008, 
        bidirectional = False,
        decay_steps = 400000,
        decay = 0.5,
        predict_step = 1000,
        max_sent_len = 30,
        uniform_init_scale = 0.1,
        clip_gradient_norm=5.0,
        save_every=25000)
    with open(path + 'paras.pkl', 'wb') as f:
        pkl.dump(paras, f)
    return paras

def train(model_path, training_data_path):
    
    with open(model_path + 'paras.pkl', 'rb') as f:
        paras = pkl.load(f)
    with open(model_path + 'vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)

    model = Skipthought_model(vocab = vocab, parameters = paras, path = model_path)
    tf.global_variables_initializer().run(session = model.sess)
    model.initialise()
    
    data_parts = glob.glob(training_data_path + '*.pkl')
    num_epochs = 1000
    model.total_loss = 0
    for epoch in range(num_epochs):
        print('\n~~~~~~~ Starting training ~~~~~~~\n')
        print('----- Epoch', epoch, '-----')
        random_permutation = np.random.permutation(len(data_parts))
        data_parts = np.array(data_parts)[random_permutation].tolist()
        for part in data_parts:
            with open(part, 'rb') as f:
                data = pkl.load(f)
            model.train(data)
        # model.save_model(model.path + '/saved_models/', model.global_step.eval(session = model.sess))
        model.epoch += 1

def continue_train(model_path, training_data_path, step):

    with open(model_path + 'paras.pkl', 'rb') as f:
        paras = pkl.load(f)
    with open(model_path + 'vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    with open(training_data_path + '/data_0.pkl', 'rb') as f:
        data = pkl.load(f)
    model = Skipthought_model(vocab = vocab, parameters = paras, path = model_path)
    model.load_model(model_path, step)
    model.initialise()
    # model.global_step = np.array([step])
    _ = model.sess.run(
                    [], 
                    feed_dict={model.global_step: step})
    
    data_parts = glob.glob(training_data_path + '*.pkl')
    num_epochs = 1000
    model.total_loss = 0
    for epoch in range(num_epochs):
        print('\n~~~~~~~ Starting training ~~~~~~~\n')
        print('----- Epoch', epoch, '-----')
        random_permutation = np.random.permutation(len(data_parts))
        data_parts = np.array(data_parts)[random_permutation].tolist()
        for part in data_parts:
            with open(part, 'rb') as f:
                data = pkl.load(f)
            model.train(data)
        # model.save_model(model.path + '/saved_models/', model.global_step.eval(session = model.sess))
        model.epoch += 1

def test(path, epoch):
    # tf.reset_default_graph()
    with open(path + 'paras.pkl', 'rb') as f:
        paras = pkl.load(f)
    with open(path + 'vocab.pkl', 'rb') as f:
        vocab = pkl.load(f)
    with open('../training_data/gingerbread/data_0.pkl', 'rb') as f:
        data = pkl.load(f)
    model = Skipthought_model(vocab = vocab, parameters = paras, path = path)
    model.load_model(path, epoch)
    model.enc_lengths, model.enc_data, model.post_data, model.post_lab, model.pre_data, model.pre_lab, model.post_masks, model.pre_masks = data[2:]
    model.evaluate(1)

if __name__ == '__main__':

    tf.reset_default_graph()
    paras = make_paras('../models/toronto_n7/')
    # preprocess(
    #     corpus_name = 'toronto', 
    #     model_path = '../models/toronto_n5/',
    #     corpus_path = '/cluster/project6/mr_corpora/vetterle/toronto_1m/', 
    #     final_path = '/cluster/project6/mr_corpora/vetterle/toronto_1m',
    #     vocab_size = 20000, 
    #     max_sent_len = paras.max_sent_len)
    train(model_path = '../models/toronto_n7/',
        training_data_path = '/cluster/project6/mr_corpora/vetterle/toronto_1m_shuffle3/')

    # paras = make_paras('../models/skipthought_gingerbread/')
    # preprocess(
    #     corpus_name = 'gingerbread', 
    #     model_path = '../models/skipthought_gingerbread/',
    #     corpus_path = '../corpus/gingerbread/', 
    #     final_path = '../training_data/gingerbread_shuffle/',
    #     vocab_size = 20000, 
    #     max_sent_len = paras.max_sent_len)
    # train(model_path = '../models/skipthought_gingerbread/',
    #     training_data_path = '../training_data/gingerbread_shuffle/')

    # continue_train(model_path = '../models/skipthought_gingerbread/',
    #     training_data_path = '../training_data/gingerbread/', step =50)


