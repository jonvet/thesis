import tensorflow as tf
import numpy as np
from util import import_data
from util import build_dictionary
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.tensorboard.plugins import projector
import time
from sys import stdout
import os
import operator
import csv


class skipthought(object):

    def __init__(self, corpus, embedding_size, hidden_size, hidden_layers, batch_size, keep_prob_dropout, learning_rate, bidirectional, loss_function, num_epochs):
        
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.batch_size = batch_size
        self.keep_prob_dropout = keep_prob_dropout
        self.learning_rate = learning_rate
        self.bidirectional = bidirectional
        self.loss_function = loss_function
        self.num_epochs = num_epochs

    def embed_data(self, data):
        return tf.nn.embedding_lookup(self.word_embeddings, data)

    def encoder(self, sentences_embedded, sentences_lengths, bidirectional = False):
        with tf.variable_scope("encoder") as varscope:
            
            
            if bidirectional:
                cell = tf.contrib.rnn.GRUCell(self.hidden_size//2)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = self.keep_prob_dropout)
                cell = tf.contrib.rnn.MultiRNNCell([cell]*self.hidden_layers, state_is_tuple=True)
                print('Training bidirectional RNN')
                sentences_outputs, sentences_states = tf.nn.bidirectional_dynamic_rnn(cell, cell, 
                    inputs = sentences_embedded, sequence_length=sentences_lengths, dtype=tf.float32)
                states_fw, states_bw = sentences_states
                sentences_states_h = tf.concat([states_fw[-1],states_bw[-1]], axis = 1)
                # sentences_states_h = tf.contrib.layers.linear(sentences_states_h, self.hidden_size)

            else:
                cell = tf.contrib.rnn.GRUCell(self.hidden_size)
                cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = self.keep_prob_dropout)
                cell = tf.contrib.rnn.MultiRNNCell([cell]*self.hidden_layers, state_is_tuple=True)
                print('Training one-directional RNN')
                sentences_outputs, sentences_states = tf.nn.dynamic_rnn(cell = cell, 
                    inputs = sentences_embedded, sequence_length=sentences_lengths, dtype=tf.float32)   
                sentences_states_h = sentences_states[-1]

        return sentences_states_h

    def decoder(self, decoder_inputs, encoder_state, name, lengths= None, train = True):

        dec_cell = tf.contrib.rnn.GRUCell(self.embedding_size)

        W = self.graph.get_tensor_by_name(name+'/weight:0')
        b = self.graph.get_tensor_by_name(name+'/bias:0')

        if train:
            with tf.variable_scope(name) as varscope:
                dynamic_fn_train = tf.contrib.seq2seq.simple_decoder_fn_train(encoder_state)
                outputs_train, state_train, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, decoder_fn = dynamic_fn_train, 
                    inputs=decoder_inputs, sequence_length = lengths, scope = varscope)
                logits = tf.reshape(outputs_train, [-1, self.embedding_size])
                logits_train = tf.matmul(logits, W) + b
                logits_projected = tf.reshape(logits_train, [self.batch_size, tf.reduce_max(lengths), self.vocabulary_size])
                return logits_projected, outputs_train

        else:
            with tf.variable_scope(name, reuse = True) as varscope:
                output_fn = lambda x: tf.nn.softmax(tf.matmul(x, W) + b)
                dynamic_fn_inference = tf.contrib.seq2seq.simple_decoder_fn_inference(output_fn =output_fn, encoder_state = encoder_state, 
                    embeddings = self.word_embeddings, start_of_sequence_id = 2, end_of_sequence_id = 3, maximum_length = self.max_sent_len, num_decoder_symbols = self.vocabulary_size) 
                logits_inference, state_inference,_ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell, decoder_fn = dynamic_fn_inference, scope = varscope)
                return tf.arg_max(logits_inference, 2)

    def get_softmax_loss(self, labels, logits):
         
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

    def get_sampled_softmax_loss(self, labels, logits, name):

        W = self.graph.get_tensor_by_name(name+'/weight:0')
        b = self.graph.get_tensor_by_name(name+'/bias:0')

        logits = tf.stack(logits)
        logits_reshaped = tf.reshape(logits, [-1, self.embedding_size])
        labels_reshaped = tf.reshape(labels, [-1, 1])
        loss = tf.nn.sampled_softmax_loss(weights= tf.transpose(W), biases=b, labels=labels_reshaped, inputs = logits_reshaped, num_sampled = 1000, 
            num_classes = self.vocabulary_size, num_true=1)
        return tf.reduce_mean(loss)

    def print_sentence(self, sentence, length):

        s = ''
        for i in range(length):
            word = sentence[i]
            s = s+self.reverse_dictionary[word]+' '
        return s

    def save_model(self, session, epoch):

        if not os.path.exists('./model/'):
            os.mkdir('./model/')
        saver = tf.train.Saver()
        if not os.path.exists('./model/epoch_%d.checkpoint' % epoch):
            saver.save(session, './model/epoch_%d.checkpoint' % epoch)
        else:
            saver.save(session, './model/epoch_%d.checkpoint' % epoch) 

    def corpus_stats(self):

        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Corpus name:', self.corpus)
        print('Vocabulary size:', len(self.dictionary))
        print('Number of sentences:', self.corpus_length)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

    def run(self):

        # Load corpus
        corpus = import_data(self.corpus)
        self.dictionary, self.reverse_dictionary, sent_lengths, self.max_sent_len, enc_data, dec_data, dec_lab = build_dictionary(corpus)

        # Save metadata for visualisation of embedding matrix
        meta_data = sorted(self.dictionary, key=model.dictionary.get)
        print(len(meta_data))
        with open('meta_data.tsv', 'w') as f:
            tsv_writer = csv.writer(f, dialect='excel')
            tsv_writer.writerow(str(i.encode('utf-8')) +'\n' for i in meta_data)


        # np.savetxt("meta_data.tsv", meta_data, fmt="%s")

        self.dictionary = sorted(self.dictionary.items(), key=operator.itemgetter(1))
        self.vocabulary_size = len(self.dictionary)
        self.max_sent_len += 1

        # Create datasets for encoder and decoders
        enc_data = enc_data[1:-1]
        enc_lengths = sent_lengths[1:-1] 
        post_lengths = sent_lengths[2:] + 1
        post_data = dec_data[2:]
        post_lab = dec_lab[2:]
        pre_lengths = sent_lengths[:-2] + 1
        pre_data = dec_data[:-2]
        pre_lab = dec_lab[:-2]
        
        # Print summary statistics
        self.corpus_length = len(enc_data)
        self.corpus_stats()


        self.graph = tf.Graph()

        with self.graph.as_default():

            print('\r~~~~~~~ Building model ~~~~~~~\r')
            self.initializer = tf.random_normal_initializer()

            # Variables
            self.word_embeddings = tf.get_variable('embeddings', [self.vocabulary_size, self.embedding_size], tf.float32, initializer = self.initializer)
            self.W_pre = tf.get_variable('precoder/weight', [self.embedding_size, self.vocabulary_size], tf.float32, initializer = self.initializer)
            self.b_pre = tf.get_variable('precoder/bias', [self.vocabulary_size], tf.float32, initializer = self.initializer)
            self.W_post = tf.get_variable('postcoder/weight', [self.embedding_size, self.vocabulary_size], tf.float32, initializer = self.initializer)
            self.b_post = tf.get_variable('postcoder/bias', [self.vocabulary_size], tf.float32, initializer = self.initializer)

            global_step = tf.Variable(0, name = 'global_step', trainable = False)

            # Encoder placeholders
            sentences = tf.placeholder(tf.int32, [None, None], "sentences")
            sentences_lengths = tf.placeholder(tf.int32, [None], "sentences_lengths")

            # Postcoder placeholders
            post_inputs = tf.placeholder(tf.int32, [None, None], "post_inputs")
            post_labels = tf.placeholder(tf.int32, [None, None], "post_labels")
            post_sentences_lengths = tf.placeholder(tf.int32, [None], "post_sentences_lengths")

            # Precoder placeholders
            pre_inputs = tf.placeholder(tf.int32, [None, None], "pre_inputs")
            pre_labels = tf.placeholder(tf.int32, [None, None], "pre_labels")
            pre_sentences_lengths = tf.placeholder(tf.int32, [None], "pre_sentences_lengths")

            # Embed sentences
            sentences_embedded = self.embed_data(sentences) 
            post_inputs_embedded = self.embed_data(post_inputs)
            pre_inputs_embedded = self.embed_data(pre_inputs)

            # Encoder
            encoded_sentences = self.encoder(sentences_embedded, sentences_lengths, self.bidirectional)

            # Decoder for following sentence
            post_logits_projected, post_logits = self.decoder(decoder_inputs = post_inputs_embedded, encoder_state = encoded_sentences, 
                name = 'postcoder', lengths = post_sentences_lengths, train = True)
            
            # Decoder for previous sentence
            pre_logits_projected, pre_logits = self.decoder(decoder_inputs = pre_inputs_embedded, encoder_state = encoded_sentences, 
                name = 'precoder', lengths = pre_sentences_lengths, train = True)
            
            # Compute loss
            if self.loss_function == 'softmax':
                post_loss = self.get_softmax_loss(post_labels, post_logits_projected) 
                pre_loss = self.get_softmax_loss(pre_labels, pre_logits_projected) 
            else:
                post_loss = self.get_sampled_softmax_loss(post_labels, post_logits, name='postcoder') 
                pre_loss = self.get_sampled_softmax_loss(pre_labels, pre_logits, name='precoder') 

            loss = pre_loss + post_loss
            opt_op = tf.contrib.layers.optimize_loss(loss = loss, global_step = global_step, learning_rate = self.learning_rate, optimizer = 'Adam', clip_gradients=2.0, 
            learning_rate_decay_fn=None, summaries = ['loss']) 

            # Decode sentences at prediction time
            pre_predict = self.decoder(decoder_inputs = pre_inputs_embedded, encoder_state = encoded_sentences, 
                name = 'precoder', lengths = pre_sentences_lengths, train = False)
            post_predict = self.decoder(decoder_inputs = post_inputs_embedded, encoder_state = encoded_sentences, 
                name = 'postcoder', lengths = post_sentences_lengths, train = False)
            predict = [pre_predict, post_predict]

        with tf.Session(graph = self.graph) as session:

            self.a= tf.contrib.graph_editor.get_tensors(self.graph)
            train_loss_writer = tf.summary.FileWriter('./tensorboard/train_loss', session.graph)


            # Use the same LOG_DIR where you stored your checkpoint.
            embedding_writer = tf.summary.FileWriter('./tensorboard/', session.graph)

            config = projector.ProjectorConfig()
            embedding = config.embeddings.add()
            embedding.tensor_name = self.word_embeddings.name
            # Link this tensor to its metadata file (e.g. labels).
            embedding.metadata_path = os.path.join('./meta_data.tsv')

            # Saves a configuration file that TensorBoard will read during startup.
            projector.visualize_embeddings(embedding_writer, config)

            merged = tf.summary.merge_all()

            print('\r~~~~~~~ Initializing variables ~~~~~~~\r')
            tf.global_variables_initializer().run()

            print('\r~~~~~~~ Starting training ~~~~~~~\r')
            start_time = time.time()
            

            try:
                train_summaryIndex = -1

                for epoch in range(self.num_epochs):
                    self.is_train = True
                    epoch_time = time.time()
                    print('----- Epoch', epoch, '-----')
                    print('Shuffling dataset')

                    perm = np.random.permutation(self.corpus_length)

                    enc_lengths_perm = enc_lengths[perm]
                    enc_data_perm = enc_data[perm]
                    post_lengths_perm = post_lengths[perm]
                    post_inputs_perm = np.array(post_data)[perm]
                    post_labels_perm = np.array(post_lab)[perm]
                    pre_lengths_perm = pre_lengths[perm]
                    pre_inputs_perm = np.array(pre_data)[perm]
                    pre_labels_perm = np.array(pre_lab)[perm]

                    total_loss = 0
                    predict_step = 10

                    for step in range(self.corpus_length // self.batch_size):

                        begin = step * self.batch_size
                        end = (step + 1) * self.batch_size

                        batch_enc_lengths = enc_lengths_perm[begin : end]
                        batch_enc_inputs = enc_data_perm[begin : end]
                        batch_post_lengths = post_lengths_perm[begin : end]
                        batch_post_inputs = post_inputs_perm[begin:end, :np.max(batch_post_lengths)]
                        batch_post_labels = post_labels_perm[begin:end, :np.max(batch_post_lengths)]
                        batch_pre_lengths = pre_lengths_perm[begin : end]
                        batch_pre_inputs = pre_inputs_perm[begin:end, :np.max(batch_pre_lengths)]
                        batch_pre_labels = pre_labels_perm[begin:end, :np.max(batch_pre_lengths)]

                        train_dict = {sentences: batch_enc_inputs, 
                                    sentences_lengths: batch_enc_lengths,
                                    post_inputs: batch_post_inputs,
                                    post_labels: batch_post_labels,
                                    post_sentences_lengths: batch_post_lengths,
                                    pre_inputs: batch_pre_inputs,
                                    pre_labels: batch_pre_labels,
                                    pre_sentences_lengths: batch_pre_lengths}

                        _, loss_val, batch_summary = session.run([opt_op, loss, merged], feed_dict=train_dict)
                        train_loss_writer.add_summary(batch_summary, step + (self.corpus_length // self.batch_size)*epoch)

                        total_loss += loss_val

                        if step % predict_step == 0:
                            if step > 0:
                                print("Average loss at step ", step, ": ", total_loss/predict_step)
                                total_loss = 0

                                print('\nOriginal sequence:\n')
                                print(self.print_sentence(batch_pre_inputs[0, 1:], batch_pre_lengths[0]-1))
                                print(self.print_sentence(batch_enc_inputs[0], batch_enc_lengths[0]))
                                print(self.print_sentence(batch_post_inputs[0, 1:], batch_post_lengths[0]-1))

                                test_enc_lengths = np.expand_dims(batch_enc_lengths[0], 0)
                                test_enc_inputs = np.expand_dims(batch_enc_inputs[0], 0)
                                test_post_lengths = np.expand_dims(batch_post_lengths[0], 0)
                                test_post_inputs = np.expand_dims(batch_post_inputs[0], 0)
                                test_post_labels = np.expand_dims(batch_post_labels[0], 0)
                                test_pre_lengths = np.expand_dims(batch_pre_lengths[0], 0)
                                test_pre_inputs = np.expand_dims(batch_pre_inputs[0], 0)
                                test_pre_labels = np.expand_dims(batch_pre_labels[0], 0)

                                test_dict = {sentences_lengths: test_enc_lengths,
                                            sentences: test_enc_inputs, 
                                            post_sentences_lengths: test_post_lengths,
                                            post_inputs: test_post_inputs,
                                            post_labels: test_post_labels,
                                            pre_sentences_lengths: test_pre_lengths,
                                            pre_inputs: test_pre_inputs,
                                            pre_labels: test_pre_labels}

                                pre_prediction, post_prediction = session.run([predict], feed_dict=test_dict)[0]

                                print('\nPredicted previous and following sequence around original sentence:\n')
                                print(self.print_sentence(pre_prediction[0], len(pre_prediction[0])))
                                print(self.print_sentence(batch_enc_inputs[0], batch_enc_lengths[0]))
                                print(self.print_sentence(post_prediction[0], len(post_prediction[0])))

                                end_time = time.time()
                                print('\nTime for 100 steps: %0.2f seconds' % (end_time - start_time))
                                start_time = time.time()
                                print('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

                    saver = tf.train.Saver()
                    saver.save(session, os.path.join('./tensorboard/', 'model.ckpt'))

            except KeyboardInterrupt:
                save = input('save?')
                if 'y' in save:
                    self.save_model(session, 0)

if __name__ == '__main__':
    tf.reset_default_graph()

    model = skipthought(corpus = 'sherlock.txt',
        embedding_size = 200, 
        hidden_size = 200, 
        hidden_layers = 1, 
        batch_size = 10, 
        keep_prob_dropout = 1.0, 
        learning_rate = 0.005, 
        bidirectional = False,
        loss_function = 'sampled_softmax',
        num_epochs = 100)

    model.run()

