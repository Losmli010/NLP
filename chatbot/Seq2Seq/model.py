import tensorflow as tf
from tensorflow.contrib import rnn

import numpy as np

import utils

def get_embeddings(word2vec_path, vocab_path, embedding_dim):
    vocab_dict = utils.load_vocab(vocab_path)
    vocab_size = len(vocab_dict)
    if word2vec_path and vocab_path:
        print("Loading word2vec embeddings...")
        word2vec_dict, embedding_dim = utils.load_words_vector(word2vec_path, vocab_dict)
        initializer = utils.build_initial_embedding_matrix(vocab_dict, word2vec_dict, embedding_dim)
    else:
        print("No word2vec path specificed, starting with random embeddings.")
        initializer = tf.random_uniform([vocab_size + 1, embedding_dim], -1.0, 1.0)

    return tf.Variable(initializer, dtype=tf.float32, name="word_embeddings")

class Seq2SeqModel(object):
    """
    Seq2Seq model for chatbot
    """
    def __init__(self, vocab_size, embedding_dim, rnn_size, num_layers, 
                 batch_size, word2vec_path, vocab_path, training):
        
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name="decoder_inputs")
        self.decoder_outputs = tf.placeholder(tf.int32, [None, None], name="decoder_outputs")
        self.input_keep_prob = tf.placeholder(tf.float32, name="input_keep_prob")
        self.output_keep_prob = tf.placeholder(tf.float32, name="output_keep_prob")
        
        #Embedding layer
        words_embedding = get_embeddings(word2vec_path, vocab_path, embedding_dim)
        encoder_embeddings = tf.nn.embedding_lookup(words_embedding, self.encoder_inputs)
        decoder_embeddings = tf.nn.embedding_lookup(words_embedding, self.decoder_inputs)
        if training:
            encoder_embeddings = tf.nn.dropout(encoder_embeddings, self.input_keep_prob)
            decoder_embeddings = tf.nn.dropout(decoder_embeddings, self.input_keep_prob)
        
        #Build lstm cell
        def lstm_cell():
            cell = rnn.LSTMCell(rnn_size, reuse=tf.get_variable_scope().reuse)
            if training:
                cell = rnn.DropoutWrapper(cell=cell, output_keep_prob=self.output_keep_prob)
            return cell
        
        #RNN for encode input    
        with tf.variable_scope("lstm") as vs:
            encoder_cells = rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
        self.initial_state = encoder_cells.zero_state(batch_size, tf.float32)
        encoder_output, encoder_last_state = tf.nn.dynamic_rnn(cell=encoder_cells, 
                                                               inputs=encoder_embeddings, 
                                                               initial_state=self.initial_state)
                
        #RNN for decode input 
        with tf.variable_scope("lstm") as vs:
            vs.reuse_variables()
            decoder_cells = rnn.MultiRNNCell(cells=[lstm_cell() for _ in range(num_layers)], state_is_tuple=True)
        decoder_output, decoder_last_state = tf.nn.dynamic_rnn(cell=decoder_cells, 
                                                               inputs=decoder_embeddings, 
                                                               initial_state=encoder_last_state)
                
        with tf.name_scope("flatten_outputs"):
            flatten_outputs = tf.reshape(decoder_output, [-1, rnn_size])
        
        with tf.name_scope("softmax"):
            weight = tf.Variable(tf.truncated_normal([rnn_size ,vocab_size], stddev=0.1) ,name="weight")
            bias = tf.Variable(tf.zeros([vocab_size]), name="bias")
            self.logits = tf.nn.bias_add(tf.matmul(flatten_outputs, weight), bias=bias, name="logits")
            self.probs = tf.nn.softmax(self.logits, name="probs")
            
        with tf.name_scope("predict"):
            self.prediction = tf.argmax(self.probs, 1, name="prediction")

        with tf.name_scope("loss"):
            labels = tf.one_hot(tf.reshape(self.decoder_outputs,[-1]), depth=vocab_size)
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=labels)
            self.loss = tf.reduce_mean(losses)
            
    def test(self, inputs):
        EOS = 1
        decoder_input_len = 5
        sess = tf.Session()
        with sess.as_default():
            sess.run(tf.global_variables_initializer())
            initial_state = sess.run(self.initial_state)
            decoder_inputs = np.array([[EOS]], dtype=np.int32)
            for _  in range(decoder_input_len): 
                feed_dict={self.encoder_inputs: inputs,
                           self.decoder_inputs: decoder_inputs,
                           self.input_keep_prob: 1.0, 
                           self.output_keep_prob: 1.0, 
                           self.initial_state: initial_state}
                predictions = sess.run(self.prediction, feed_dict=feed_dict)
                next_word = np.array([[predictions[-1]]])
                decoder_inputs = np.hstack((decoder_inputs, next_word))
            print("Inputs is %s" %inputs)
            print("Prediction is %s" %predictions)
            
def generate_sequence(length):
    seq = np.zeros((1,length * 2), dtype=np.int32)
    x = np.random.randint(length * 2, size=length)
    seq[:,length:] = x
    return seq