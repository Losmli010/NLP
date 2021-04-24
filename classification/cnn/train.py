# coding: utf-8

import os
import datetime
import argparse

import numpy as np
import tensorflow as tf

import utils
from model import TextCNN


def hparams():
    parser = argparse.ArgumentParser()
    
    # Data loading params
    parser.add_argument("--dev_sample_percentage", type=float, default=0.3, help="Percentage of the training data to use for validation")
    parser.add_argument("--positive_sentiment_file", type=str, default="data/positive_sentiment.txt", help="Data source for the positive data")
    parser.add_argument("--negative_sentiment_file", type=str, default="data/negative_sentiment.txt", help="Data source for the negative data")
    parser.add_argument("--vocab_file", type=str, default="data/vocab.txt", help="Vocabulary")
    parser.add_argument("--word2vec_file", type=str, default=None, help="Chinese wiki word2vec")
    
    # Model Parameters
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimensionality of the word embedding")
    parser.add_argument("--filter_sizes", type=str, default="3,4,5", help="Comma-separated filter sizes")
    parser.add_argument("--num_filters", type=int, default=128, help="Number of filters per filter size")
    
    # Training Parameters
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of time steps")
    parser.add_argument("--dropout_keep_prob", type=float, default=0.5, help="Dropout keep probability")
    parser.add_argument("--l2_lambda", type=float, default=1e-4, help="Lambda for l2 regularization")
    parser.add_argument("--evaluate_steps", type=int, default=100, help="Evaluate model on dev set after this many steps")

    args, _ = parser.parse_known_args()
    return args


def train():
    args = hparams()
    
    # Load data
    print("Loading data...")
    x_text, y = utils.load_data(args.positive_sentiment_file, args.negative_sentiment_file)

    # Build vocabulary
    max_sentence_length = max([len(x.split()) for x in x_text])
    utils.get_vocab(args.vocab_file, x_text)
    vocab, vocab_dict = utils.load_vocab(args.vocab_file)
    x = np.array(utils.fit_transform(x_text, vocab_dict))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    dev_sample_index = -1 * int(args.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    
    # Training
    sess = tf.Session()
    with sess.as_default():
        cnn = TextCNN(sequence_length=max_sentence_length,
                      num_classes=2,
                      vocab_size=len(vocab),
                      embedding_dim=args.embedding_dim,
                      filter_sizes=list(map(int, args.filter_sizes.split(","))),
                      num_filters=args.num_filters, 
                      l2_lambda=args.l2_lambda, 
                      word2vec_path=args.word2vec_file, 
                      vocab_path=args.vocab_file)
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        
        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", cnn.loss)
        acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.abspath(os.path.join(os.path.curdir, "summaries", "train"))
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.abspath(os.path.join(os.path.curdir, "summaries", "dev"))
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        
        #Save model params
        checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, "model")
        saver = tf.train.Saver(tf.global_variables())
        
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Generate batches
        batches = utils.batch_iter(list(zip(x_train, y_train)), args.batch_size, args.num_epochs)
        
        dev_accuracy = 0.0
        # Training loop...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            feed_dict = {cnn.inputs: x_batch, 
                         cnn.targets: y_batch, 
                         cnn.dropout_keep_prob: args.dropout_keep_prob}
            _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy], 
                                               feed_dict=feed_dict)
            
            train_summary_writer.add_summary(summaries, step)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, accuracy))
            
            # Evaluate on dev set
            current_step = tf.train.global_step(sess, global_step)
            if current_step % args.evaluate_steps == 0:
                dev_feed_dict = {cnn.inputs: x_dev, 
                                 cnn.targets: y_dev, 
                                 cnn.dropout_keep_prob: 1.0}
                step, summaries, loss, accuracy = sess.run([global_step, dev_summary_op, cnn.loss, cnn.accuracy], 
                                                feed_dict=dev_feed_dict)
                
                dev_summary_writer.add_summary(summaries, step)
                time_str = datetime.datetime.now().isoformat()
                print("Evaluate on dev set:")
                print("{}: step {}, loss {:g}, accuracy {:g}".format(time_str, step, loss, accuracy))
                
                if dev_accuracy < accuracy:
                    dev_accuracy = accuracy
                    saver.save(sess, checkpoint_file, global_step=current_step)
                    print("Save better model to %s" % checkpoint_file)
