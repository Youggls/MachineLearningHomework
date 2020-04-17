from nltk import PorterStemmer
import numpy as np
import scipy.io as scio
import re

def load_voca(path):
    file = open(path, mode='r')
    voca_dic = {}
    for line in file.readlines():
        line = line.strip('\n')
        index, word = line.split('\t')
        voca_dic[word] = int(index)
    return voca_dic

def split(delimiters, string, maxsplit=0):
    pattern = '|'.join(map(re.escape, delimiters))
    return re.split(pattern, string, maxsplit)

def preprocess_email(email_content, voca):
    email_content = email_content.lower()
    email_content = re.sub('<[^<>]+>', ' ', email_content)
    email_content = re.sub('[0-9]+', 'number', email_content)
    email_content = re.sub('(http|https)://[^\s]*', 'httpaddr', email_content)
    email_content = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_content)
    email_content = re.sub('[$]+', 'dollar', email_content)
    words = split(""" @$/#.-:&*+=[]?!(){},'">_<;%\n\r""", email_content)
    word_indices = []
    stemmer = PorterStemmer()
    for word in words:
        word = re.sub('[^a-zA-Z0-9]', '', word)
        if word == '':
            continue
        word = stemmer.stem(word)
        print(word, end=' ')
        if word in voca:
            idx = voca[word]
            word_indices.append(idx)

    return word_indices

def read_matlab(path):
    return scio.loadmat(path)

def email_features(word_indices):
    # Total number of words in the dictionary
    n = 1899

    x = np.zeros((n, 1))
    x[word_indices] = 1

    return x
