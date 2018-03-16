# coding: utf-8

import os
import argparse

import tensorflow as tf
import numpy as np

import utils

def hparams():
    parser = argparse.ArgumentParser()
    
    # Data loading params
    parser.add_argument("--positive_sentiment_file", type=str, default="data/positive_sentiment.txt", help="Data source for the positive data")
    parser.add_argument("--negative_sentiment_file", type=str, default="data/negative_sentiment.txt", help="Data source for the negative data")
    parser.add_argument("--vocab_file", type=str, default="data/vocab.txt", help="Vocabulary")
    
    # Model Parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")

    args,_ = parser.parse_known_args()
    return args

def test():
    args = hparams()
    
    x_raw, y_test = utils.load_data(args.positive_sentiment_file, args.negative_sentiment_file)
    y_test = np.argmax(y_test, axis=1)

    # Map data into vocabulary
    vocab, vocab_dict = utils.load_vocab(args.vocab_file)
    x_test = np.array(utils.fit_transform(x_raw, vocab_dict))

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
            # input_y = graph.get_operation_by_name("input_y").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/prediction").outputs[0]
        
            # Generate batches for one epoch
            batches = utils.batch_iter(list(x_test), args.batch_size, 1, shuffle=False)

            # Collect the predictions here
            all_predictions = []

            for batch in batches:
                batch_predictions = sess.run(predictions, feed_dict={inputs: batch, dropout_keep_prob: 1.0})
                all_predictions = np.concatenate([all_predictions, batch_predictions])

    # Print accuracy if y_test is defined
    correct_predictions = float(sum(all_predictions == y_test))
    print("Total number of test examples: {}".format(len(y_test)))
    print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))
