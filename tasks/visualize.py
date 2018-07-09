import sys
import tensorflow as tf
import numpy as np
import pickle as pkl 
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import seaborn as sn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from infersent.data import get_nli
from skipthought.skipthought import Skipthought_model
from infersent.infersent import Infersent_model

cluster = False
CBOW = True
_learning_rate = 0.0001
_batch_size = 64
_epochs = 10
_dropout = 0.9

if cluster:
    SKIPTHOUGHT_PATH = '/home/vetterle/skipthought/code/'
    MODEL_PATH = '/cluster/project2/mr/vetterle/skipthought/toronto_n5/'
    SKIPTHOUGHT_PATH = '/cluster/project2/mr/vetterle/skipthought/toronto_n5/'
    INFERSENT_PATH = '/home/vetterle/InferSent/code/'
    SICK_PATH = '/home/vetterle/skipthought/eval/SICK/'
    SNLI_PATH = '/home/vetterle/InferSent/dataset/SNLI/'
    TORONTO_PATH = '/cluster/project6/mr_corpora/vetterle/toronto/'
    SAVE_PATH = '/cluster/project2/mr/vetterle/thesis'

else:    
    MODEL_PATH = '/Users/Jonas/Documents/Repositories/skipthought/models/toronto_n5/'
    SKIPTHOUGHT_PATH = '/Users/Jonas/Documents/Repositories/skipthought/models/toronto_n5/'
    INFERSENT_PATH = '/Users/Jonas/Documents/Repositories/InferSent/code/'
    SICK_PATH = '/Users/Jonas/Documents/Repositories/skipthought/eval/SICK/'
    SNLI_PATH = '/Users/Jonas/Documents/Repositories/InferSent/dataset/SNLI/'
    TORONTO_PATH = '/Users/Jonas/Documents/Repositories/skipthought/corpus/'
    SAVE_PATH = '..'



MODELS = ['skipthought', 'infersent']
MODEL = MODELS[0]

print('Loading corpus')
train, dev, test = get_nli(SNLI_PATH)
train = np.array(train['s2'])
dev = np.array(dev['s2'])
test = np.array(test['s2'])

print('Loading saved model')
tf.reset_default_graph()
embeddings = None # in case of 'cbow' or 'infersent' model
n_iter = 0

with open(MODEL_PATH + 'vocab.pkl', 'rb') as f:
    vocab = pkl.load(f)

if MODEL == 'skipthought':
    step = 488268
    with open(MODEL_PATH + 'paras.pkl', 'rb') as f:
        paras = pkl.load(f)
    model = Skipthought_model(vocab = vocab, parameters = paras, path = MODEL_PATH)
    model.load_model(MODEL_PATH, step)

    dictionary = defaultdict(int)
    embeddings = np.load(MODEL_PATH + 'expanded_embeddings.npy')
    with open(MODEL_PATH + 'expanded_vocab.pkl', 'rb') as f:
        expanded_vocab = pkl.load(f)

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
    model.vocab = vocab
    SENT_DIM = 620 if CBOW else 2400
    WORD_DIM = 620
    embeddings = None

elif MODEL == 'infersent':
    # step = 77247
    step = 128745
    with open(MODEL_PATH + 'paras.pkl', 'rb') as f:
        paras = pkl.load(f)
    model = Infersent_model(vocab = vocab, parameters = paras, path = MODEL_PATH)
    model.load_model(MODEL_PATH, step)
    model.para.batch_size = _batch_size
    SENT_DIM = 300 if CBOW else 4096
    WORD_DIM = 300

print('%s model loaded' %MODEL)

def embed(sentences):

    sentences = [[w for w in sentence.split(' ')] for sentence in sentences]
    sentences_embedded, sentences_lengths = [], []

    if MODEL == 'skipthought':
        sentences_lengths = np.array([len(sentence) for sentence in sentences], dtype=np.int32)
        batch_embedded = np.full([len(sentences), np.max(sentences_lengths), 620], vocab['<PAD>'], dtype=np.float32)
        for i, sentence in enumerate(sentences):
            words = [vocab[word] if word in vocab else vocab['<OOV>'] for word in sentence]
            batch_embedded[i, :len(sentence), :] = np.array(words)
        if CBOW:
            sentences_embedded = np.mean(batch_embedded, axis = 1)
        else:
            test_dict = {model.sentences_embedded: batch_embedded, 
            model.sentences_lengths: sentences_lengths}
            sentences_embedded = model.encoded_sentences.eval(session = model.sess, feed_dict=test_dict)
        return np.array(sentences_embedded)

    else:
        for i in range(len(sentences)):
            s_f = [word for word in sentences[i] if word in vocab]
            if not s_f:
                s_f = ['</s>']
            sentences[i] = s_f
        if CBOW:
            batch_words = [np.array([vocab[word] for word in sentence]) for sentence in sentences]
            batch_embedded = [np.mean(sentence, axis = 0) for sentence in batch_words]
            batch_lengths = [len(sentence) for sentence in sentences]
            sentences_embedded.append(batch_embedded)
            sentences_lengths.append(batch_lengths)

        else:
            batch_s, batch_l = model.get_batch(sentences)
            test_dict = {
                model.s1_embedded: batch_s,
                model.s1_lengths: batch_l}
            batch_embedded = model.sess.run(
                model.s1_states_h, 
                feed_dict=test_dict)
            sentences_embedded.append(batch_embedded)
            sentences_lengths.append(batch_l)
        return np.squeeze(sentences_embedded)

cbow = 'CBOW' if CBOW else ''
if MODEL == 'infersent':
    x = 64
    y = 64
    if CBOW:
        x = 15
        y = 20
elif MODEL == 'skipthought':
    x = 48
    y = 50
    if CBOW:
        x = 20
        y = 31

# x = 64 if MODEL=='infersent' else 48
# y = 64 if MODEL=='infersent' else 50

sentences_m = ['He walked down the alley',
    'The cook sharpened his knife',
    'Paul rings the door bell',
    'Adam eats a sandwich for lunch',
    'Rock music is not his favourite',
    'Once in a while, a man comes to buy flowers' ,
    'The school boy forgot his books',
    'A man is sitting near a bike and is writing a note ',
    'Two men are taking a break from a trip on a snowy road',
    'A boy is standing outside the water ',
    'A few men in a competition are running outside',
    'A man is playing the flute',
    'Some dangerous men with knives are throwing a bike against a tree ',
    'A man is climbing a rope',
    'There is no man drawing',
    'A guy is cutting some ginger',
    'Chris is a keen swimmer',
    'Mr May is a poor guy',
    'The king found a way to the tower',
    'The Duke eats chicken',
    'The Little Baron is a movie',
    'The Emperor watches the sun set',
    'Everyone deserves a good brother',
    'My big brother is a student',
    'Male students study in the library',
    'The librarian writes with his pen',
    'The children and their uncles are going to the kindergarden',
    'Uncles are brothers of your parents',
    'The parents have a son',
    'The son has a cat',
    'Gentlemen are smoking cigars',
    'Dave drinks a coffee in the cafe',
    'The prince is melking cows',
    'Stewards are working on a plane',
    'The host welcomes the guests',
    'The waiter brings the bill',
    'In some schools the headmaster is not very popular',
    'In order to secure safety, policemen are waiting outside',
    'The chairman introduces new colleagues',
    'In Hollywood, there are many actors']





sentences_f = ['She walked down the alley',
    'The cook sharpened her knife',
    'Paula rings the door bell',
    'Eva eats a sandwich for lunch',
    'Rock music is not her favourite',
    'Once in a while, a woman comes to buy flowers' ,
    'The school girl forgot her books',
    'A woman is sitting near a bike and is writing a note ',
    'Two women are taking a break from a trip on a snowy road',
    'A girl is standing outside the water ',
    'A few women in a competition are running outside',
    'A woman is playing the flute',
    'Some dangerous women with knives are throwing a bike against a tree ',
    'A woman is climbing a rope',
    'There is no woman drawing',
    'A girl is cutting some ginger',
    'Kate is a keen swimmer',
    'Mrs May is a poor girl',
    'The queen found a way to the tower',
    'The Duchess eats chicken',
    'The Little Baroness is a movie',
    'The Empress watches the sun set',
    'Everyone deserves a good sister',
    'My big sister is a student',
    'Female students study in the library',
    'The librarian writes with her pen',
    'The children and their aunts are going to the kindergarden',
    'Aunts are sisters of your parents',
    'The parents have a daughter',
    'The daughter has a cat',
    'Ladies are smoking cigars',
    'Dana drinks a coffee in the cafe',
    'The princess is melking cows',
    'Stewardesses are working on a plane',
    'The hostess welcomes the guests',
    'The waitress brings the bill',
    'In some schools the headmistress is not very popular',
    'In order to secure safety, policewomen are waiting outside',
    'The chairwoman introduces new colleagues',
    'In Hollywood, there are many actresses']

embedded_m = embed(sentences_m)
embedded_f = embed(sentences_f)

embedded_m = np.reshape(embedded_m,[-1,x,y])
embedded_f = np.reshape(embedded_f,[-1,x,y])

embedded_m_std = np.std(embedded_m,0)
embedded_f_std = np.std(embedded_f,0)

embedded_m = np.mean(embedded_m,0)
embedded_f = np.mean(embedded_f,0)

embedded_m_f_se = np.sqrt(np.square(embedded_m_std)/40 + np.square(embedded_f_std)/40)
# df = (np.square(embedded_m_std)/40 + np.square(embedded_f_std)/40) / (np.square(np.square(embedded_m_std)/40)/39 + np.square(np.square(embedded_f_std)/40)/(39))
df = 39
t_value = 2.0227
t_m_f = (embedded_m-embedded_f)/embedded_m_f_se
t_m_f_sig = (t_m_f<-t_value) + (t_m_f>t_value)

embedded_m = (embedded_m-np.min(embedded_m))/(np.max(embedded_m)-np.min(embedded_m))
embedded_f = (embedded_f-np.min(embedded_f))/(np.max(embedded_f)-np.min(embedded_f))

plt.close()
sn.heatmap(embedded_m, xticklabels=False, yticklabels=False, cbar=True, vmin=-1, vmax=1, cmap='RdBu_r')
plt.savefig('../plots/%s%sm.png' % (MODEL,cbow))
plt.close()
sn.heatmap(embedded_f, xticklabels=False, yticklabels=False, cbar=True, vmin=-1, vmax=1, cmap='RdBu_r')
plt.savefig('../plots/%s%sf.png' % (MODEL,cbow))
plt.close()
sn.heatmap(embedded_m-embedded_f, xticklabels=False, yticklabels=False, cbar=True, vmin=-1.0, vmax=1.0, cmap='RdBu_r')
plt.savefig('../plots/%s%sm-f.png' % (MODEL,cbow))
plt.close()
sn.heatmap(t_m_f_sig, xticklabels=False, yticklabels=False, cbar=True, vmin=-1.0, vmax=1.0, cmap='RdBu_r')
plt.savefig('../plots/t_%s%sm-f.png' % (MODEL,cbow))
plt.close()



sentences_present = ['He walks down the alley',
    'The cook sharpens his knife',
    'Paul rings the door bell',
    'Adam eats a sandwich for lunch',
    'Rock music is not his favourite',
    'Once in a while, a man comes to buy flowers' ,
    'The school boy forgets his books',
    'A man is sitting near a bike and is writing a note ',
    'Two men are taking a break from a trip on a snowy road',
    'A boy is standing outside the water ',
    'A few men in a competition are running outside',
    'A man is playing flute',
    'Some dangerous women with knives are throwing a bike against a tree ',
    'A woman is climbing a rope',
    'There is no woman drawing',
    'A girl is cutting some ginger',
    'Kate is a keen swimmer',
    'Mrs May is a poor girl',
    'The queen finds a way to the tower',
    'The Duchess eats chicken',
    'The Little Baroness is a movie',
    'The Empress watches the sun set',
    'The police man prepares his uniform',
    'The car drives down the street',
    'President Trump is lying',
    'Ranger Smith lights the fire',
    'The camp fire glows in the dark',
    'Dark rooms have no light',
    'Light weight boxers fight',
    'Fighting turtles are popular',
    'Popular singers sell many CDs',
    'Music industry bosses earn a lot of money',
    'Money does not make you happy',
    'Happy hippies dance in trance',
    'Trance artists use synthetic instruments',
    'Instruments lie on the floor in the concert hall',
    'Hall publishes economic articles',
    'Articles in the newspaper contain useful information',
    'Information leaks mean problems for the agents',
    'Chemical agents react with others']



sentences_past = ['He walked down the alley',
    'The cook sharpened his knife',
    'Paul rang the door bell',
    'Adam ate a sandwich for lunch',
    'Rock music was not his favourite',
    'Once in a while, a man came to buy flowers' ,
    'The school boy forgot his books',
    'A man was sitting near a bike and is writing a note ',
    'Two men were taking a break from a trip on a snowy road',
    'A boy was standing outside the water ',
    'A few men in a competition were running outside',
    'A man was playing flute',
    'Some dangerous women with knives were throwing a bike against a tree ',
    'A woman was climbing a rope',
    'There was no woman drawing',
    'A girl was cutting some ginger',
    'Kate was a keen swimmer',
    'Mrs May was a poor girl',
    'The queen found a way to the tower',
    'The Duchess ate chicken',
    'The Little Baroness was a movie',
    'The Empress watched the sun set',
    'The police man prepared his uniform',
    'The car drove down the street',
    'President Trump was lying',
    'Ranger Smith lit the fire',
    'The camp fire glew in the dark',
    'Dark rooms had no light',
    'Light weight boxers fought',
    'Fighting turtles were popular',
    'Popular singers sold many CDs',
    'Music industry bosses earned a lot of money',
    'Money did not make you happy',
    'Happy hippies danced in trance',
    'Trance artists used synthetic instruments',
    'Instruments lay on the floor in the concert hall',
    'Hall published economic articles',
    'Articles in the newspaper contained useful information',
    'Information leaked mean problems for the agents',
    'Chemical agents reacted with others']

embedded_present = embed(sentences_present)
embedded_past = embed(sentences_past)

embedded_present = np.reshape(embedded_present,[-1,x,y])
embedded_past = np.reshape(embedded_past,[-1,x,y])

embedded_present_std = np.std(embedded_present,0)
embedded_past_std = np.std(embedded_past,0)

embedded_present = np.mean(embedded_present,0)
embedded_past = np.mean(embedded_past,0)

embedded_present_past_se = np.sqrt(np.square(embedded_present_std)/40 + np.square(embedded_past_std)/40)
# df = (np.square(embedded_m_std)/40 + np.square(embedded_f_std)/40) / (np.square(np.square(embedded_m_std)/40)/39 + np.square(np.square(embedded_f_std)/40)/(39))
df = 39
t_value = 2.0227
t_present_past = (embedded_present-embedded_past)/embedded_present_past_se
t_present_past_sig = (t_present_past<-t_value) + (t_present_past>t_value)

embedded_present = (embedded_present-np.min(embedded_present))/(np.max(embedded_present)-np.min(embedded_present))
embedded_past = (embedded_past-np.min(embedded_past))/(np.max(embedded_past)-np.min(embedded_past))


plt.close()
sn.heatmap(embedded_present, xticklabels=False, yticklabels=False, cbar=True, vmin=-1, vmax=1, cmap='RdBu_r')
plt.savefig('../plots/%s%spresent.png' % (MODEL,cbow))
plt.close()
sn.heatmap(embedded_past, xticklabels=False, yticklabels=False, cbar=True, vmin=-1, vmax=1, cmap='RdBu_r')
plt.savefig('../plots/%s%spast.png' % (MODEL,cbow))
plt.close()
sn.heatmap(embedded_present-embedded_past, xticklabels=False, yticklabels=False, cbar=True, vmin=-1, vmax=1, cmap='RdBu_r')
plt.savefig('../plots/%s%spresent-past.png' % (MODEL,cbow))
plt.close()
sn.heatmap(t_present_past_sig, xticklabels=False, yticklabels=False, cbar=True, vmin=-1.0, vmax=1.0, cmap='RdBu_r')
plt.savefig('../plots/t_%s%spresent-past.png' % (MODEL,cbow))
plt.close()

sentences_singular = ['He walked down the alley',
    'The cook sharpened his knife',
    'Paul rings the door bell',
    'Adam eats a sandwich for lunch',
    'Rock music is not his favourite',
    'Once in a while, a man comes to buy flowers' ,
    'The school boy forgot his books',
    'A man is sitting near a bike and is writing a note ',
    'One man is taking a break from a trip on a snowy road',
    'A boy is standing outside the water ',
    'One man in a competition is running outside',
    'A man is playing the flute',
    'A dangerous man with knives is throwing a bike against a tree ',
    'A man is climbing a rope',
    'There is no man drawing',
    'A guy is cutting some ginger',
    'Chris is a keen swimmer',
    'Mr May is a poor guy',
    'The king found a way to the tower',
    'The Duke eats chicken',
    'The Little Baron is a movie',
    'The Emperor watches the sun set',
    'Everyone deserves a good brother',
    'My big brother is a student',
    'A male student studies in the library',
    'The librarian writes with his pen',
    'The children and their uncle are going to the kindergarden',
    'An uncle is a brother of your parent',
    'The parent has a son',
    'The son has a cat',
    'A Gentleman are smoking cigars',
    'Dave drinks a coffee in the cafe',
    'The prince is melking cows',
    'A steward is working on a plane',
    'The host welcomes the guests',
    'The waiter brings the bill',
    'In some schools the headmaster is not very popular',
    'In order to secure safety, a policeman is waiting outside',
    'The chairman introduces a new colleague',
    'In Hollywood, there is an actor']

sentences_plural = ['They walked down the alley',
    'The cooks sharpened their knives',
    'Paul and Paula ring the door bell',
    'Adam and Eve eat sandwiches for lunch',
    'Rock music is not their favourite',
    'Once in a while, two men come to buy flowers' ,
    'The school boys forgot their books',
    'Two men are sitting near a bike and are writing notes ',
    'Two men are taking a break from a trip on a snowy road',
    'Some boys are standing outside the water ',
    'A few men in a competition are running outside',
    'Five men are playing the flute',
    'Some dangerous men with knives are throwing a bike against a tree ',
    'A man is climbing a rope',
    'There are no men drawing',
    'The guys are cutting some ginger',
    'Chris and Ken are keen swimmers',
    'Mr May and Mr April are poor guys',
    'The kings found a way to the tower',
    'The Dukes eat chicken',
    'The Little Barons is a movie',
    'The Emperors watch the sun set',
    'Everyone deserves good brothers',
    'My big brothers are students',
    'Male students study in the library',
    'The librarians write with their pens',
    'The children and their uncles are going to the kindergarden',
    'Uncles are brothers of your parents',
    'The parents have sons',
    'The sos have cats',
    'Gentlemen are smoking cigars',
    'Dave and David drink a coffee in the cafe',
    'The princes are melking cows',
    'Stewards are working on a plane',
    'The hosts welcome the guests',
    'The waiters bring the bills',
    'In some schools the headmasters are not very popular',
    'In order to secure safety, policemen are waiting outside',
    'The chairmen introduce new colleagues',
    'In Hollywood, there are many actors']

embedded_singular = embed(sentences_singular)
embedded_plural = embed(sentences_plural)

embedded_singular = np.reshape(embedded_singular,[-1,x,y])
embedded_plural = np.reshape(embedded_plural,[-1,x,y])

embedded_singular_std = np.std(embedded_singular,0)
embedded_plural_std = np.std(embedded_plural,0)

embedded_singular = np.mean(embedded_singular,0)
embedded_plural = np.mean(embedded_plural,0)

embedded_singular_plural_se = np.sqrt(np.square(embedded_singular_std)/40 + np.square(embedded_plural_std)/40)
# df = (np.square(embedded_m_std)/40 + np.square(embedded_f_std)/40) / (np.square(np.square(embedded_m_std)/40)/39 + np.square(np.square(embedded_f_std)/40)/(39))
df = 39
t_value = 2.0227
t_singular_plural = (embedded_singular-embedded_plural)/embedded_singular_plural_se
t_singular_plural_sig = (t_singular_plural<-t_value) + (t_singular_plural>t_value)


embedded_singular = (embedded_singular-np.min(embedded_singular))/(np.max(embedded_singular)-np.min(embedded_singular))
embedded_plural = (embedded_plural-np.min(embedded_plural))/(np.max(embedded_plural)-np.min(embedded_plural))


plt.close()
sn.heatmap(embedded_singular, xticklabels=False, yticklabels=False, cbar=True, vmin=-1, vmax=1, cmap='RdBu_r')
plt.savefig('../plots/%s%ssingular.png' % (MODEL,cbow))
plt.close()
sn.heatmap(embedded_plural, xticklabels=False, yticklabels=False, cbar=True, vmin=-1, vmax=1, cmap='RdBu_r')
plt.savefig('../plots/%s%splural.png' % (MODEL,cbow))
plt.close()
sn.heatmap(embedded_singular-embedded_plural, xticklabels=False, yticklabels=False, cbar=True, vmin=-1, vmax=1, cmap='RdBu_r')
plt.savefig('../plots/%s%ssingular-plural.png' % (MODEL,cbow))
plt.close()
sn.heatmap(t_singular_plural_sig, xticklabels=False, yticklabels=False, cbar=True, vmin=-1.0, vmax=1.0, cmap='RdBu_r')
plt.savefig('../plots/t_%s%ssingular-plural.png' % (MODEL,cbow))
plt.close()


print(np.sum(t_m_f_sig)/(x*y))
print(np.sum(t_present_past_sig)/(x*y))
print(np.sum(t_singular_plural_sig)/(x*y))

print(np.mean(abs(t_m_f[t_m_f_sig==1])))
print(np.mean(abs(t_present_past[t_present_past_sig==1])))
print(np.mean(abs(t_singular_plural[t_singular_plural_sig==1])))
