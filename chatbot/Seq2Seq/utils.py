# coding: utf-8

from collections import Counter

import numpy as np

EOS = "EOS"
UNK = "UNK"
VOCAB_SIZE = 50000


def load_data(filename):
    x_input = []
    y_input = []
    y_output = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split("||")
            x_input.append(line[0])
            y_input.append("{} {}".format(EOS, line[1]))
            y_output.append("{} {}".format(line[1], EOS))
    return x_input, y_input, y_output


def get_vocab(filename, text):
    counter = Counter()
    for sent in text:
        counter += Counter(sent.split())
    vocab = counter.most_common(VOCAB_SIZE)
    with open(filename, "w", encoding="utf-8") as f:
        for w, _ in vocab:
            f.write(w + "\n")
    print("Save vocab to %s" %filename)


def load_vocab(filename):
    with open(filename, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
        vocab.append(UNK)
        vocab_dict = dict(zip(vocab, range(1, len(vocab) + 1)))
    return vocab_dict


def encoder_transform(text, vocab_dict):
    max_sentence_length = max([len(x.split()) for x in text])
    text2num = []
    for sent in text:
        sent2num = []
        for word in sent.split():
            if word in vocab_dict:
                sent2num.append(vocab_dict[word])
            else:
                sent2num.append(vocab_dict[UNK])
        pad_length = max_sentence_length - len(sent2num)
        sent2num = list(np.zeros(pad_length, dtype=np.int32)) + sent2num
        text2num.append(sent2num)
    return text2num


def decoder_transform(text, vocab_dict):
    max_sentence_length = max([len(x.split()) for x in text])
    text2num = []
    for sent in text:
        sent2num = []
        for word in sent.split():
            if word in vocab_dict:
                sent2num.append(vocab_dict[word])
            else:
                sent2num.append(vocab_dict[UNK])
        pad_length = max_sentence_length - len(sent2num)
        sent2num += list(np.zeros(pad_length, dtype=np.int32))
        text2num.append(sent2num)
    return text2num


def load_words_vector(filename, vocab_dict):
    word2vec_dict = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            vector = tokens[1:]
            if word in vocab_dict:
                word2vec_dict[word] = [float(x) for x in vector]
    embedding_dim = len(vector)
    print("Found {} out of {} in word2vec." .format(len(word2vec_dict), len(vocab)))
    return word2vec_dict,embedding_dim


def build_initial_embedding_matrix(vocab_dict, word2vec_dict, embedding_dim):
    initial_embeddings = np.random.uniform(-1.0, 1.0, (len(vocab_dict) + 1, embedding_dim))
    for word, index in vocab_dict.items():
        if word in word2vec_dict:
            initial_embeddings[index, :] = word2vec_dict[word]
    return initial_embeddings


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches = int(data_size / batch_size)
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
