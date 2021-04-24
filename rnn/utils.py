# coding: utf-8

import pickle
import numpy as np


def load_data(text_file):
    f = open(text_file, "r", encoding="utf-8")
    sentences = [line.strip() for line in f]
    print("Parsed %d sentences." % (len(sentences)))
    f.close()
    
    mean_sentence_length = int(sum([len(sent.split()) for sent in sentences])/len(sentences))
    print("Mean of sentences length is %s" % mean_sentence_length)
    
    text = " ".join(sentences)
    return text.split()


def build_vocab(text):
    vocab = list(set(text))
    print("There are %s unique tokens" % len(vocab))
    vocab_dict = dict(zip(vocab, range(len(vocab))))
    return vocab_dict


def save_vocab(vocab_dict ,vocab_file):
    with open(vocab_file, "wb") as f:
        pickle.dump(vocab_dict, f)
    f.close()
    print("Save vocab to %s" % vocab_file)


def load_vocab(vocab_file):
    with open(vocab_file, "rb") as f:
        vocab_dict = pickle.load(f)
    f.close()
    print("Load vocab from %s" % vocab_file)
    return vocab_dict


def transform(text, vocab_dict):
    return [vocab_dict[word] for word in text]


def batch_iter(data, sequence_length, batch_size):
    data = np.array(data)
    data_size = len(data)
    num_batches = int(data_size / (sequence_length * batch_size))
    for batch_num in range(num_batches):
        start_index = batch_num * sequence_length * batch_size
        end_index = (batch_num + 1) * sequence_length * batch_size
        yield data[start_index:end_index].reshape((batch_size, sequence_length))
