import numpy as np
import pickle as pkl
import glob

# input_path = '../training_data/gingerbread/'
# output_path = '../training_data/gingerbread_shuffle/'
input_path = '/cluster/project6/mr_corpora/vetterle/toronto_new/'
output_path = '/cluster/project6/mr_corpora/vetterle/toronto_shuffle/'

files  = glob.glob(input_path + '*.pkl')
number = 40

print('\nLoading...')
with open(files[0], 'rb') as f:
    print(files[0])
    file = pkl.load(f)
    enc_lengths, enc_data, post_data, post_lab, pre_data, pre_lab, post_masks, pre_masks = file[2:]
    corpus_name = file[0]
    max_len = file[1]

for file_path in files[1:]:
    with open(file_path, 'rb') as f:
        print(file_path)
        t_enc_lengths, t_enc_data, t_post_data, t_post_lab, t_pre_data, t_pre_lab, t_post_masks, t_pre_masks = file[2:]
        
        enc_lengths = np.concatenate((enc_lengths, t_enc_lengths), axis = 0)
        enc_data = np.concatenate((enc_data, t_enc_data), axis = 0)
        post_data = np.concatenate((post_data, t_post_data), axis = 0)
        post_lab = np.concatenate((post_lab, t_post_lab), axis = 0)
        pre_data = np.concatenate((pre_data, t_pre_data), axis = 0)
        pre_lab = np.concatenate((pre_lab, t_pre_lab), axis = 0)
        post_masks = np.concatenate((post_masks, t_post_masks), axis = 0)
        pre_masks = np.concatenate((pre_masks, t_pre_masks), axis = 0)

print('\nShuffling..')
n = len(enc_lengths)
perm = np.random.permutation(n)

enc_lengths = enc_lengths[perm]
enc_data = enc_data[perm]
post_data = post_data[perm]
post_lab = post_lab[perm]
pre_data = pre_data[perm]
pre_lab = pre_lab[perm]
post_masks = post_masks[perm]
pre_masks = pre_masks[perm]


b_size = n // 5 + 1
print('\nSaving...')
for x in range(5):
    with open(output_path + ('data_%d.pkl' % x), 'wb') as f:
        print(output_path + ('data_%d.pkl' % x))
        t_enc_lengths = enc_lengths[x*b_size : (x+1)*b_size]
        t_enc_data = enc_data[x*b_size : (x+1)*b_size, :]
        t_post_data = post_data[x*b_size : (x+1)*b_size, :]
        t_post_lab = post_lab[x*b_size : (x+1)*b_size, :]
        t_pre_data = pre_data[x*b_size : (x+1)*b_size, :]
        t_pre_lab = pre_lab[x*b_size : (x+1)*b_size, :]
        t_post_masks = post_masks[x*b_size : (x+1)*b_size, :]
        t_pre_masks = pre_masks[x*b_size : (x+1)*b_size, :]

        pkl.dump([corpus_name, max_len, t_enc_lengths, t_enc_data, t_post_data, t_post_lab, t_pre_data, t_pre_lab, t_post_masks, t_pre_masks], f)




