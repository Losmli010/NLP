# coding: utf-8

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np

class RNNLM(object):
    def __init__(self, vocab_size, embedding_dim, rnn_size, 
                 num_layers, batch_size, training):
        
        self.input_data = tf.placeholder(tf.int64, [None, None], name="inputs")
        self.targets = tf.placeholder(tf.int64, [None, None], name="targets")
        self.input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
        
        #Embedding layer
        with tf.name_scope("embedding"):
            words_embedding = tf.Variable(
                tf.random_uniform([vocab_size, embedding_dim], -1.0, 1.0), name="words_embedding")
            inputs = tf.nn.embedding_lookup(words_embedding, self.input_data)
            if training:
                inputs = tf.nn.dropout(inputs, self.input_keep_prob)
        
        #Build LSTM cell
        def lstm_cell():
            cell = rnn.LSTMCell(rnn_size, state_is_tuple=True)
            if training:
                cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=self.output_keep_prob)
            return cell
                
        stacked_cells = rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
        self.initial_state = stacked_cells.zero_state(batch_size, tf.float32)
        outputs, last_state = tf.nn.dynamic_rnn(cell=stacked_cells, 
                                                     inputs=inputs, 
                                                     initial_state=self.initial_state)

        with tf.name_scope("flatten_outputs"):
            flatten_outputs = tf.reshape(outputs, [-1, rnn_size])
        
        with tf.name_scope("softmax"):
            weight = tf.Variable(tf.truncated_normal([rnn_size ,vocab_size], stddev=0.1) ,name="weight")
            bias = tf.Variable(tf.zeros([vocab_size]), name="bias")
            self.logits = tf.nn.bias_add(tf.matmul(flatten_outputs, weight), bias=bias, name="logits")
            self.probs = tf.nn.softmax(self.logits, name="probs")
            
        with tf.name_scope("output"):
            self.prediction = tf.argmax(self.probs, 1, name="prediction")

        with tf.name_scope("loss"):
            labels = tf.one_hot(tf.reshape(self.targets,[-1]), depth=vocab_size)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels)
            self.loss = tf.reduce_mean(losses)