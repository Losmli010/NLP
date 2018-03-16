# coding: utf-8

import os
import argparse

import tensorflow as tf
import numpy as np
import jieba

from model import Seq2SeqModel
import utils

EOS = "EOS"

def generate(query):
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_file", type=str, default="data/vocab.txt", help="Vocabulary dictionary")
    parser.add_argument("--word2vec_file", type=str, default=None, help="Chinese wiki word2vec")
    parser.add_argument("--vocab_size", type=int, default=50001, help="Vocabulary size")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimensionality of the words embedding")
    parser.add_argument("--rnn_size", type=int, default=256, help="Hidden units of rnn layer ")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of rnn layer")
    parser.add_argument("--batch_size", type=int, default=1, help="Minibatch size")
    args,_ = parser.parse_known_args()
    
    vocab_dict = utils.load_vocab(args.vocab_file)
    index2word = dict(zip(vocab_dict.values(), vocab_dict.keys()))

    query = list(jieba.cut(query))
    query2id = np.array(utils.encoder_transform(query, vocab_dict))
    query2id = query2id.reshape((1, len(query)))

    checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session()
        with sess.as_default():
            model = Seq2SeqModel(vocab_size=args.vocab_size, 
                             embedding_dim=args.embedding_dim, 
                             rnn_size=args.rnn_size, 
                             num_layers=args.num_layers, 
                             batch_size=args.batch_size, 
                             word2vec_path=args.word2vec_file, 
                             vocab_path=args.vocab_file, 
                             training=False)
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_file)
            
            decoder_inputs = np.array([[vocab_dict[EOS]]], dtype=np.int32)
            predictions = [-1]
            while predictions[-1] != vocab_dict[EOS]: 
                feed_dict={model.encoder_inputs: query2id, model.decoder_inputs: decoder_inputs}
                predictions = sess.run(model.prediction, feed_dict=feed_dict)
                next_word = np.array([[predictions[-1]]])
                decoder_inputs = np.hstack((decoder_inputs, next_word))
                if len(predictions) > 100:
                    break
                
    answer = [index2word[index] for index in predictions][:-1]
    return "".join(answer)