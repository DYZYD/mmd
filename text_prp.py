import os
import pickle
import gensim
doc_path = r'./textsss/'
w2v_path = r'C:\BaiduNetdiskDownload\GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(w2v_path, binary=True)
print("load success\n")

def load_binary_vec(fname, vocab):
    word_vecs = {}
    with open(fname, 'rb') as fin:
        header = fin.readline()

d = {}
for file in os.listdir(doc_path):
    #print(file)
    sentenses = []
    for line in open(doc_path+file, 'r'):
        line = line.split('_DELIM_')[-1].strip()
        sentence = []
        for word in line.split():
            try:
                if len(sentence) >= 50:
                    break
                sentence.append(model[word].tolist())
            except:
                ...
        length = len(sentence)
        if length < 50:
            for i in range(50 - length):
                sentence.append([0 for k in range(300)])
        sentenses.append(sentence)
    d[file.split('.')[0]] = sentenses

with open('text_w2v_raw.pickle', 'wb') as file:
    pickle.dump(d, file)
    print('extract features success')
