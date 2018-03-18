import numpy as np
import pickle as pkl
import glob

input_path = '/cluster/project6/mr_corpora/vetterle/toronto_new/'
# output_path = '/cluster/project6/mr_corpora/vetterle/toronto_1m_shuffle4/'
output_path = '../training_data/'

files  = glob.glob(input_path + '*.pkl')
n_files = len(files)
perm = np.random.permutation(n_files)
files = np.array(files)[perm]
b_size = 9

print('To do: %d batches' % (n_files // b_size + 1))
for batch in range(n_files // b_size + 1):
    print('\nBatch %d' % (batch+1))

    batch_files = files[batch*b_size : (batch+1)*b_size]
    print('Files:', batch_files)

    with open(batch_files[0], 'rb') as f:
        file = pkl.load(f)
        print(file)
        enc_lengths, enc_data, post_data, post_lab, pre_data, pre_lab, post_masks, pre_masks = file[2:]
        corpus_name = file[0]
        max_len = file[1]

        for file_path in batch_files[1:]:
            with open(file_path, 'rb') as f:
                t_enc_lengths, t_enc_data, t_post_data, t_post_lab, t_pre_data, t_pre_lab, t_post_masks, t_pre_masks = file[2:]
                
                enc_lengths = np.concatenate((enc_lengths, t_enc_lengths), axis = 0)
                enc_data = np.concatenate((enc_data, t_enc_data), axis = 0)
                post_data = np.concatenate((post_data, t_post_data), axis = 0)
                post_lab = np.concatenate((post_lab, t_post_lab), axis = 0)
                pre_data = np.concatenate((pre_data, t_pre_data), axis = 0)
                pre_lab = np.concatenate((pre_lab, t_pre_lab), axis = 0)
                post_masks = np.concatenate((post_masks, t_post_masks), axis = 0)
                pre_masks = np.concatenate((pre_masks, t_pre_masks), axis = 0)

    print('Shuffling..')
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

    lines_per_file = 1000000
    n_output_files = n // lines_per_file + 1
    print('Number of output files:', n_output_files)
    print('Saving...')
    for x in range(n_output_files):
        n = len(glob.glob(output_path + '*.pkl'))
        with open(output_path + 'data_%d.pkl' % n, 'wb') as f:
            t_enc_lengths = enc_lengths[x*lines_per_file : (x+1)*lines_per_file]
            t_enc_data = enc_data[x*lines_per_file : (x+1)*lines_per_file, :]
            t_post_data = post_data[x*lines_per_file : (x+1)*lines_per_file, :]
            t_post_lab = post_lab[x*lines_per_file : (x+1)*lines_per_file, :]
            t_pre_data = pre_data[x*lines_per_file : (x+1)*lines_per_file, :]
            t_pre_lab = pre_lab[x*lines_per_file : (x+1)*lines_per_file, :]
            t_post_masks = post_masks[x*lines_per_file : (x+1)*lines_per_file, :]
            t_pre_masks = pre_masks[x*lines_per_file : (x+1)*lines_per_file, :]

            pkl.dump([corpus_name, max_len, t_enc_lengths, t_enc_data, t_post_data, t_post_lab, t_pre_data, t_pre_lab, t_post_masks, t_pre_masks], f)




