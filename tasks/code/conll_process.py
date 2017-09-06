import pickle as pkl

file = 'test'

sentences = []
sentence = []
i = 0
with open('../preprocess_data/%s_s2.txt.conll' % file) as f:
# with open('../preprocess_data/dev.txt.conll') as f:
    for line in f:
        data = line.split()
        # print(data)
        if data == []:
            sentences.append(sentence)
            sentence = []
            i+=1
        else: 
            sentence.append(data)
print(i)
data = []
for sentence in sentences:
    sent = []
    # print(sentence)
    for word in sentence:
        # print(word)
        obs = []
        obs.append(word[1])
        dependent = int(word[5]) - 1
        obs.append(sentence[dependent][1])
        obs.append(word[6])
        sent.append(obs)
    data.append(sent)

with open('../preprocess_data/%s_s2.pkl' %file, 'wb') as f:
    pkl.dump(data, f)