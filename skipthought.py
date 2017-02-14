# TODO: dynamic decoder
# TODO: negative sampling
# TODO: second decoder for preceding sentences


import nltk
import tensorflow as tf
import numpy as np
import collections
import math
from tensorflow.python.ops import variable_scope

def import_data(path):

    '''
    Imports data and tokenises data.
    Outputs a list of sentences, where each sentence is itself a list of words
    '''

    sentences = []
    words = []
    with open(path, 'r') as myfile:
        data = myfile.read().replace('\n', ' ')
    sentences = nltk.sent_tokenize(data)
    for sentence in sentences:
        this_sentence = nltk.word_tokenize(sentence)
        words.append(this_sentence)
    return words

def build_dictionary(sentences, vocab=None, max_sent_len_=None):

    is_ext_vocab = True

    # If no vocab provided, create a new one
    if vocab is None:
        is_ext_vocab = False
        vocab = {'<PAD>': 0, '<OOV>': 1, '<GO>': 2, '<END>': 3}

    # Create list that will contain integer encoded senteces 
    data_sentences = []
    max_sent_len = -1

    for sentence in sentences:
        words = []
        for word in sentence:
            # If creating a new vocab, and word isnt in vocab yet, add it
            if not is_ext_vocab and word not in vocab:
                vocab[word] = len(vocab)
            # Now add either OOV or actual token to words
            if word not in vocab:
                token_id = vocab['<OOV>']
            else:
                token_id = vocab[word]
            words.append(token_id)
        if len(words) > max_sent_len:
            max_sent_len = len(words)
        data_sentences.append(words)

    if max_sent_len_ is not None:
        max_sent_len = max_sent_len_

    enc_sentences = np.full([len(data_sentences), max_sent_len], vocab['<PAD>'], dtype=np.int32)
    dec_go_sentences = np.full([len(data_sentences), max_sent_len + 1], vocab['<PAD>'], dtype=np.int32)
    dec_end_sentences = np.full([len(data_sentences), max_sent_len + 1], vocab['<PAD>'], dtype=np.int32)

    # Create inputs for encoder, which will all be as long as the max_sent_length in the corpus, and padded
    # Also create inputs and labels for decoder. These will be 1 token longer to allow for GO and END tokens.
    sentence_lengths = []
    for i, sentence in enumerate(data_sentences):
        sentence_lengths.append(len(sentence))
        enc_sentences[i, 0:len(sentence)] = sentence
        dec_go_sentences[i, 0] = vocab['<GO>']
        dec_go_sentences[i, 1:len(sentence)+1] = sentence
        dec_end_sentences[i, 0:len(sentence)] = sentence
        dec_end_sentences[i, len(sentence)] = vocab['<END>']

    sentence_lengths = np.array(sentence_lengths, dtype=np.int32)
    reverse_dictionary = dict(zip(vocab.values(), vocab.keys()))

    # enc_sentences = enc_sentences[:,:5]
    # dec_go_sentences = dec_go_sentences[:,:5]
    # dec_end_sentences = dec_end_sentences[:,:5]
    # max_sent_len = 4

    return vocab, reverse_dictionary, sentence_lengths, max_sent_len+1, enc_sentences, dec_go_sentences, dec_end_sentences

def loop_function(a, b):

    a_unpacked = [tf.reshape(tensor, [1, _embedding_size]) for tensor in tf.unpack(a, axis=0)]
    logits_projected = []
    for logit in a_unpacked:
        logit_projected = tf.contrib.layers.linear(logit, _vocabulary_size) 
        this_max = tf.argmax(logit_projected, 1)
        logits_projected.append(this_max)

    a = tf.pack(logits_projected)
    a = tf.nn.embedding_lookup(word_embeddings, a)
    a = tf.reshape(a, [_num_test, _embedding_size])

    return a


if __name__ == '__main__':

    tf.reset_default_graph()
    _embedding_size = 200
    _hidden_size = 200
    _num_steps = 20000
    _hidden_layers = 3
    _batch_size = 10
    _keep_prob_dropout = 1.0
    _file_path = 'gingerbread.txt'
    _corpus = import_data(_file_path)
    _dictionary, _reverse_dictionary, _sentence_lengths, _max_sent_len, _enc_inputs, _dec_inputs, _dec_labels = build_dictionary(_corpus)
    _vocabulary_size = len(_dictionary)
    _num_test = 3
    learning_rate = 0.001

    # Temporary: align enoder and decoder inputs
    _enc_inputs = _enc_inputs[1:-1]
    _dec_inputs = _dec_inputs[2:]
    _dec_labels = _dec_labels[2:]

    # Placeholders
    sentences = tf.placeholder(tf.int64, [None, None], "sentences") # [batch_size, max_sent_length]
    sentences_lengths = tf.placeholder(tf.int64, [None], "sentences_lengths")
    dec_inputs = tf.placeholder(tf.int64, [None, None], "dec_inputs")
    dec_labels = tf.placeholder(tf.int64, [None, _max_sent_len], "dec_labels")
    gold = tf.placeholder(tf.int32, [None, 3], "gold") # [batch_size * 3] contains triplets corresponding to batch number, number of word in sequence and index in vocab

    # Variables
    word_embeddings = tf.get_variable('embeddings', [_vocabulary_size, _embedding_size], tf.float32, initializer=tf.random_normal_initializer())

    # Embedded sentences
    sentences_embedded = tf.nn.embedding_lookup(word_embeddings, sentences) # [batch_size, max_sent_length, embedding_size]
    dec_inputs_embedded = tf.nn.embedding_lookup(word_embeddings, dec_inputs) # [batch_size, max_sent_length, embedding_size]


    #-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  
    #-  -  -  -  -  -  -  -  -  -  -  -  E N C O D E R  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -

    cell = tf.nn.rnn_cell.GRUCell(_hidden_size)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=_keep_prob_dropout)
    cell = tf.nn.rnn_cell.MultiRNNCell([cell]*_hidden_layers, state_is_tuple=True)

    with tf.variable_scope("encoder") as varscope:
        sentences_outputs, sentences_states = tf.nn.dynamic_rnn(cell = cell, inputs = sentences_embedded, sequence_length=sentences_lengths, dtype=tf.float32)   
        sentences_states_h = sentences_states[-1]
        # sentences_states_h = sentences_outputs[-1]


    #-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  
    #-  -  -  -  -  -  -  -  -  -  -  -  D E C O D E R  -  T R A I N I N G   -  -  -  -  -  -  -  -  -  -  -  - 

    dec_cell = tf.nn.rnn_cell.GRUCell(_embedding_size)

    # Decoder needs lists as input
    dec_inputs_list = [tf.reshape(x,[_batch_size, _embedding_size]) for x in tf.split(1,_max_sent_len, dec_inputs_embedded)]

    # At training time feed the decoder the gold inputs
    with tf.variable_scope("decoder") as varscope:
    	dec_out, dec_state = tf.nn.seq2seq.rnn_decoder(dec_inputs_list, sentences_states_h, dec_cell, scope=varscope)
    logits = [tensor for tensor in tf.unpack(dec_out, axis=1)]

    # Project output of decoder onto something of shape vocabulary_size
    logits_projected = []
    with tf.variable_scope("decoder/loop_function") as varscope:
        for logit in logits:
        	logit_projected = tf.contrib.layers.linear(logit, _vocabulary_size) 
        	logits_projected.append(logit_projected)

    probabilities = tf.nn.softmax(logits_projected) # [batch_size, max_sent_length, vocab]
    probabilities_gold = tf.gather_nd(probabilities, gold) # [batch_size * max_sent_length]
    log_probabilities_gold = tf.log(probabilities_gold)
    loss = -tf.reduce_sum(log_probabilities_gold)
    opt_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # optimizer = tf.train.AdamOptimizer(learning_rate)
    # gvs = optimizer.compute_gradients(loss)
    # grad_norms = [tf.nn.l2_loss(g) for g, v in gvs]
    # grad_norm = tf.add_n(grad_norms)
    # capped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gvs]
    # opt_op = optimizer.apply_gradients(capped_gvs)

    #-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  
    #-  -  -  -  -  -  -  -  -  -  -  -  D E C O D E R  -  T E S T I N G  -  -  -  -  -  -  -  -  -  -  -  -  -


    pred_inputs_list = [tf.reshape(x,[_num_test, _embedding_size]) for x in tf.split(1,_max_sent_len, dec_inputs_embedded)]

    with tf.variable_scope("decoder", reuse=True) as varscope:
        pred_out, _ = tf.nn.seq2seq.rnn_decoder(pred_inputs_list, sentences_states_h, dec_cell, loop_function, scope=varscope)

    pred_logits = [tensor for tensor in tf.unpack(pred_out, axis=1)]

    pred_logits_projected = []
    with tf.variable_scope("decoder/loop_function", reuse=True) as varscope:
        for pred_logit in pred_logits:
            pred_logit_projected = tf.contrib.layers.linear(pred_logit, _vocabulary_size)
            pred_logits_projected.append(pred_logit_projected)

    pred_probabilities = tf.nn.softmax(pred_logits_projected)
    predict = tf.arg_max(pred_probabilities, 2)

    with tf.Session() as session:

        tf.global_variables_initializer().run()
        print("Initialized")
        average_loss = 0

        for step in range(1,_num_steps):
            total_loss = 0
            # print('Shuffling dataset')
            perm = np.random.permutation(len(_enc_inputs))[:_batch_size] 

            batch_enc_inputs = _enc_inputs[perm]
            batch_inputs_length = _sentence_lengths[perm]
            batch_dec_inputs = np.array(_dec_inputs)[perm]
            batch_dec_labels = np.array(_dec_labels)[perm]

            batch_labels = []
            for row in range(_batch_size):
                for column in range(_max_sent_len):
                    batch_labels.append([row, column,batch_dec_labels[row, column]]) 

            batch_labels = np.array(batch_labels)


            feed_dict = {sentences: batch_enc_inputs, 
                        sentences_lengths: batch_inputs_length,
                        dec_inputs: batch_dec_inputs,
                        dec_labels: batch_dec_labels,
                        gold: batch_labels}
            _, loss_val, probs, probsgold = session.run([opt_op, loss, probabilities, probabilities_gold], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 100 == 0:
                if step > 0:
                    average_loss /= 100
                    print("Average loss at step ", step, ": ", average_loss)
                    average_loss = 0

                    batch_enc_inputs = batch_enc_inputs[:_num_test]
                    batch_inputs_length = batch_inputs_length[:_num_test]
                    batch_dec_inputs = batch_dec_inputs[:_num_test]
                    batch_dec_labels = batch_dec_labels[:_num_test]

                    for sentence in range(_num_test):
                        s = ''
                        for word in batch_dec_inputs[sentence]:
                            s=s+_reverse_dictionary[word]+' '
                        print('Gold:', s)

                    feed_dict = {sentences: batch_enc_inputs, 
                                sentences_lengths: batch_inputs_length,
                                dec_inputs: batch_dec_inputs,
                                dec_labels: batch_dec_labels}
                    prediction = session.run([predict], feed_dict=feed_dict)[0]

                    for sentence in range(_num_test):
                        s = ''
                        for word in prediction[sentence]:
                            s=s+_reverse_dictionary[word]+' '
                        print('Prediction:', s)
