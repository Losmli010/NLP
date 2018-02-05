# coding: utf-8

import numpy as np
import re

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data(positive_sentiment_file, negative_sentiment_file):
    positive_sentiments = [line.strip() for line in open(positive_sentiment_file, "r", encoding="utf-8")]
    negative_sentiments = [line.strip() for line in open(negative_sentiment_file, "r", encoding="utf-8")]
    x_text = positive_sentiments + negative_sentiments
    x_text = [clean_str(sent) for sent in x_text]
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