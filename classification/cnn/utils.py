# coding: utf-8

import numpy as np
import re


def clean_str(string):
    string = re.sub(r"[^\u4E00-\u9FA5，！？。\“\”《》A-Za-z0-9(),!?\'\`]", " ", string)
    return string.strip()


def load_data(positive_sentiment_file, negative_sentiment_file):
    positive_sentiments = [line.strip() for line in open(positive_sentiment_file, "r", encoding="utf-8")]
    negative_sentiments = [line.strip() for line in open(negative_sentiment_file, "r", encoding="utf-8")]
    x_text = positive_sentiments + negative_sentiments
    positive_labels = [[0, 1] for _ in positive_sentiments]
    negative_labels = [[1, 0] for _ in negative_sentiments]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches = int((data_size - 1)/batch_size) + 1
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


def get_vocab(filename, text):
    words = [word for sent in text for word in sent.split()]
    vocab = set(words)
    with open(filename, "w", encoding="utf-8") as f:
        for w in vocab:
            f.write(w + "\n")
    print("Save vocab to %s" %filename)
    

def load_vocab(filename):
    with open(filename, "r", encoding="utf-8") as f:
        vocab = [line.strip() for line in f]
        vocab_dict = dict(zip(vocab, range(len(vocab))))
    return vocab, vocab_dict


def fit_transform(text, vocab_dict):
    max_sentence_length = max([len(x.split()) for x in text])
    text2num = []
    for sent in text:
        sent2num = [vocab_dict[word] for word in sent.split()]
        pooled_length = max_sentence_length - len(sent2num)
        sent2num += list(np.zeros((pooled_length)))
        text2num.append(sent2num)
    return text2num


def load_words_vector(filename, vocab):
    word2vec_dict = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            tokens = line.strip().split()
            word = tokens[0]
            vector = tokens[1:]
            if word in vocab:
                word2vec_dict[word] = [float(x) for x in vector]
    embedding_dim = len(vector)
    print("Found {} out of {} in word2vec." .format(len(word2vec_dict), len(vocab)))
    return word2vec_dict,embedding_dim


def build_initial_embedding_matrix(vocab_dict, word2vec_dict, embedding_dim):
    initial_embeddings = np.random.uniform(-1.0, 1.0, (len(vocab_dict), embedding_dim))
    for word, index in vocab_dict.items():
        if word in word2vec_dict:
            initial_embeddings[index, :] = word2vec_dict[word]
    return initial_embeddings
