import tensorflow as tf
from skipthought.skipthought import Skipthought_para
from skipthought.skipthought import Skipthought_model
from skipthought.skipthought import gru_cell
from infersent.infersent import Infersent_para
from infersent.infersent import Infersent_model
from infersent.data import get_nli
import pickle as pkl
import numpy as np
from collections import defaultdict


class Encoder(object):

    def __init__(self, model_name, model_path, snli_path, untrained=False, cbow=False):
        self.model_name = model_name
        self.untrained = untrained
        self.cbow = cbow
        self.word_dim_dict = {'infersent': 300, 'skipthought': 620}
        self.sent_dim_dict = {'infersent': {False: 4096, True: 300}, 'skipthought': {False: 2400, True: 620}}
        self.word_dim = self.word_dim_dict[model_name]
        self.sent_dim = self.sent_dim_dict[model_name][cbow]

        print('Loading saved model')
        tf.reset_default_graph()
        embeddings = None # in case of 'cbow' or 'infersent' model

        with open(model_path + 'vocab.pkl', 'rb') as f:
            vocab = pkl.load(f)

        if model_name == 'skipthought':

            step = 488268
            with open(model_path + 'paras.pkl', 'rb') as f:
                paras = pkl.load(f)
            self.model = Skipthought_model(vocab = vocab, parameters = paras, path = model_path)
            if untrained:
                self.model.sess = tf.Session(graph = self.model.graph)
                tf.global_variables_initializer().run(session = self.model.sess)
            else:
                self.model.load_model(model_path, step)

            dictionary = defaultdict(int)
            embeddings = np.load(model_path + 'expanded_embeddings.npy')
            with open(model_path + 'expanded_vocab.pkl', 'rb') as f:
                expanded_vocab = pkl.load(f)

            print('Loading corpus')
            train, dev, test = get_nli(snli_path)
            train = np.array(train['s2'])
            dev = np.array(dev['s2'])
            test = np.array(test['s2'])

            for part in [train, dev, test]:
                for i in range(len(part)):
                    sentence = part[i].split()
                    for word in sentence:
                        dictionary[word] += 1
            vocab = {}
            for word in ['<OOV>','<PAD>']:
                vocab[word] = embeddings[expanded_vocab[word]]
            for word in dictionary.keys():
                try:
                    vocab[word] = embeddings[expanded_vocab[word]]
                except:
                    next
            print('Found {0}(/{1}) words with word2vec vectors'.format(len(vocab), len(expanded_vocab)))
            self.model.vocab = vocab

            embeddings = None

        elif model_name == 'infersent':

            # step = 77247
            step = 128745
            with open(model_path + 'paras.pkl', 'rb') as f:
                paras = pkl.load(f)
            self.model = Infersent_model(vocab = vocab, parameters = paras, path = model_path)
            if untrained:
                self.model.sess = tf.Session(graph = self.model.graph)
                tf.global_variables_initializer().run(session = self.model.sess)
            else:
                self.model.load_model(model_path, step)
            # self.model.para.batch_size = self.batch_size

        print('{} model loaded'.format(self.model))



    def embed(self, sentences):

        sentences = [[w for w in sentence.split(' ')] for sentence in sentences]
        sentences_embedded, sentences_lengths = [], []

        if self.model_name == 'skipthought':
            sentences_lengths = np.array([len(sentence) for sentence in sentences], dtype=np.int32)
            batch_embedded = np.full([len(sentences), np.max(sentences_lengths), 620], self.model.vocab['<PAD>'], dtype=np.float32)
            for i, sentence in enumerate(sentences):
                words = [self.model.vocab[word] if word in self.model.vocab else self.model.vocab['<OOV>'] for word in sentence]
                batch_embedded[i, :len(sentence), :] = np.array(words)
            if self.cbow:
                sentences_embedded = np.mean(batch_embedded, axis = 1)
            else:
                test_dict = {self.model.sentences_embedded: batch_embedded, 
                self.model.sentences_lengths: sentences_lengths,
                self.model.keep_prob_dropout: 1.0}
                sentences_embedded = self.model.encoded_sentences.eval(
                    session = self.model.sess, 
                    feed_dict=test_dict)
            return (sentences_embedded, sentences_lengths)

        elif self.model_name == 'infersent':
            for i in range(len(sentences)):
                s_f = [word for word in sentences[i] if word in self.model.vocab]
                if not s_f:
                    s_f = ['</s>']
                sentences[i] = s_f
            if self.cbow:
                batch_words = [np.array([self.model.vocab[word] for word in sentence]) for sentence in sentences]
                batch_embedded = [np.mean(sentence, axis = 0) for sentence in batch_words]
                batch_lengths = [len(sentence) for sentence in sentences]
                sentences_embedded.append(batch_embedded)
                sentences_lengths.append(batch_lengths)

            else:
                batch_s, batch_l = self.model.get_batch(sentences)
                test_dict = {
                    self.model.s1_embedded: batch_s,
                    self.model.s1_lengths: batch_l}
                batch_embedded = self.model.sess.run(
                    self.model.s1_states_h, 
                    feed_dict=test_dict)
                sentences_embedded.append(batch_embedded)
                sentences_lengths.append(batch_l)
            return (np.squeeze(sentences_embedded), np.squeeze(sentences_lengths))
