# coding: utf-8

import os
import argparse

import tensorflow as tf
import numpy as np

import utils
from model import RNNLM

def test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, default="data/vocab.pkl", help="Vocabulary dictionary")
    parser.add_argument("--text_file", type=str, default="data/littleprince.txt", help="Data source for the shakespeare data")
    parser.add_argument("--sequence_length", type=int, default=25, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    args,_ = parser.parse_known_args()
    
    vocab_dict = utils.load_vocab(args.vocab_file)
    index2word = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    text = utils.load_data(args.text_file)
    data = utils.transform(text, vocab_dict)

    # Evaluation
    checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            inputs = graph.get_operation_by_name("inputs").outputs[0]
            input_keep_prob = graph.get_operation_by_name("input_keep_prob").outputs[0]
            output_keep_prob = graph.get_operation_by_name("output_keep_prob").outputs[0]
            predictions = graph.get_operation_by_name("output/prediction").outputs[0]
            
            all_predictions = []
            batches = utils.batch_iter(data, args.sequence_length, args.batch_size)
            for batch in batches:
                feed_dict={inputs: batch, 
                           input_keep_prob: 1.0, 
                           output_keep_prob: 1.0}
                batch_predictions = sess.run(predictions, feed_dict=feed_dict)
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                
    content = [index2word[prediction] for prediction in all_predictions]
    return "".join(content) 

def generate(start_word, length):
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, default="data/vocab.pkl", help="Vocabulary dictionary")
    parser.add_argument("--vocab_size", type=int, default=2854, help="Vocabulary size")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimensionality of the words embedding")
    parser.add_argument("--rnn_size", type=int, default=128, help="Hidden units of rnn layer ")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of rnn layer")
    parser.add_argument("--batch_size", type=int, default=1, help="Minibatch size")
    args,_ = parser.parse_known_args()
    
    vocab_dict = utils.load_vocab(args.vocab_file)
    index2word = dict(zip(vocab_dict.values(), vocab_dict.keys()))

    text = [start_word]
    text_data = utils.transform(text, vocab_dict)

    checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            rnn = RNNLM(vocab_size=args.vocab_size, 
                        embedding_dim=args.embedding_dim, 
                        rnn_size=args.rnn_size, 
                        num_layers=args.num_layers, 
                        batch_size=args.batch_size, 
                        training=False)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, checkpoint_file)
            
            for _ in range(length):
                data = np.array([text_data])
                data.reshape((1,len(text_data)))
                predictions = sess.run(rnn.prediction, feed_dict={rnn.input_data: data})
                text_data.append(predictions[-1])
                
    content = [index2word[index] for index in text_data]
    return "".join(content)