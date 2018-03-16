# coding: utf-8

import tensorflow as tf
import numpy as np

import utils

def get_embeddings(word2vec_path, vocab_path, embedding_dim):
    vocab, vocab_dict = utils.load_vocab(vocab_path)
    vocab_size = len(vocab)
    if word2vec_path and vocab_path:
        print("Loading word2vec embeddings...")
        word2vec_dict, embedding_dim = utils.load_words_vector(word2vec_path, vocab)
        initializer = utils.build_initial_embedding_matrix(vocab_dict, word2vec_dict, embedding_dim)
    else:
        print("No word2vec path specificed, starting with random embeddings.")
        initializer = tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0)

    return tf.Variable(initializer, dtype=tf.float32, name="word_embeddings")

class TextCNN(object):
    """
    A CNN for text classification
    """
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_dim, 
                 filter_sizes, num_filters, l2_lambda, word2vec_path, vocab_path):
        
        self.inputs = tf.placeholder(tf.int32, [None, sequence_length], name="inputs")
        self.targets = tf.placeholder(tf.float32, [None, num_classes], name="targets")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        l2_loss = tf.constant(0.0)
        
        with tf.device("/cpu:0"), tf.name_scope("embedding"):
            self.words_embedding = get_embeddings(word2vec_path, vocab_path, embedding_dim)
            self.embedded_words = tf.nn.embedding_lookup(self.words_embedding, self.inputs)
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)
        
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                filter_shape = [filter_size, embedding_dim, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(self.embedded_words_expanded, W, strides=[1,1,1,1], padding="VALID", name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h, ksize=[1, sequence_length - filter_size + 1, 1, 1], strides=[1,1,1,1], padding="VALID", name="pool")
                pooled_outputs.append(pooled)
                
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            
        with tf.name_scope("output"):
            W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.prediction = tf.argmax(self.scores, 1, name="prediction")
            
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.targets)
            self.loss = tf.reduce_mean(losses) + l2_lambda * l2_loss
            
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.prediction, tf.argmax(self.targets, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
