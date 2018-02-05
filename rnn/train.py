# coding: utf-8

import os
import datetime
import argparse

import numpy as np
import tensorflow as tf

import utils
from model import RNNLM

def hparams():
    parser = argparse.ArgumentParser()
    
    #Data loading params
    parser.add_argument("--dev_sample_percentage", type=float, default=0.1, help="Percentage of the training data to use for validation")
    parser.add_argument("--text_file", type=str, default="data/littleprince.txt", help="Data source for the shakespeare data")
    parser.add_argument("--vocab_file", type=str, default="data/vocab.pkl", help="Vocabulary dictionary")
    
    # Model Parameters
    parser.add_argument("--vocab_size", type=int, default=2854, help="Vocabulary size")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimensionality of the words embedding")
    parser.add_argument("--rnn_size", type=int, default=128, help="Hidden units of rnn layer ")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of rnn layer")
    
    # Training Parameters
    parser.add_argument("--sequence_length", type=int, default=25, help="Sequence length")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--num_epochs", type=int, default=2000, help="Number of iterations")
    parser.add_argument("--input_keep_prob", type=float, default=0.5, help="Dropout rate of embedding layer")
    parser.add_argument('--output_keep_prob', type=float, default=0.5, help="Dropout rate of rnn layer")
    parser.add_argument("--max_grad_norm", type=float, default=5.0, help="Clip global grad norm")
    parser.add_argument("--checkpoint_steps", type=int, default=500, help="Save model after this many steps")
    parser.add_argument("--evaluate_steps", type=int, default=500, help="Evaluate model on dev set after this many steps")

    args,_ = parser.parse_known_args()  
    return args

def train():
    args = hparams()
    
    print("Load data...")
    x_text = utils.load_data(args.text_file)
    vocab_dict = utils.build_vocab(x_text)
    x_data = utils.transform(x_text, vocab_dict)
    
    x_data = x_data[:-2]
    y_data = x_data[1:]
    
    # Split train/test set
    dev_sample_index = -1 * int(args.dev_sample_percentage * float(len(x_data)))
    x_train, x_dev = x_data[:dev_sample_index], x_data[dev_sample_index:]
    y_train, y_dev = y_data[:dev_sample_index], y_data[dev_sample_index:]
    print("Train/Dev split: {:d}/{:d}".format(len(x_train), len(x_dev)))
    
    utils.save_vocab(vocab_dict, args.vocab_file)
    del x_text, x_data, y_data
    
    #Training
    sess = tf.Session()
    with sess.as_default():
        rnn = RNNLM(vocab_size=args.vocab_size, 
                    embedding_dim=args.embedding_dim, 
                    rnn_size=args.rnn_size, 
                    num_layers=args.num_layers, 
                    batch_size=args.batch_size, 
                    training=True)
        
        #Define train_op
        global_step = tf.Variable(0, name="global_step", trainable=False)
        learning_rate = tf.Variable(args.learning_rate, name="learning_rate", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(rnn.loss, tvars), args.max_grad_norm)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        
        #Save model params
        checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, "model")
        
        #Save best model params
        dev_dir = os.path.abspath(os.path.join(os.path.curdir, "dev"))
        if not os.path.exists(dev_dir):
            os.makedirs(dev_dir)
        dev_file = os.path.join(dev_dir, "model")
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
        
        dev_loss = 2e+50
        
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Training loop...
        for epoch in range(args.num_epochs):
            # Generate batches
            x_batches = utils.batch_iter(x_train, args.sequence_length, args.batch_size)
            y_batches = utils.batch_iter(y_train, args.sequence_length, args.batch_size)
            initial_state = sess.run(rnn.initial_state)
            for x_batch, y_batch in zip(x_batches, y_batches):
                feed_dict = {rnn.input_data: x_batch, 
                             rnn.targets: y_batch, 
                             rnn.input_keep_prob: args.input_keep_prob, 
                             rnn.output_keep_prob: args.output_keep_prob, 
                             rnn.initial_state: initial_state}
                _, step, loss = sess.run([train_op, global_step, rnn.loss], feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
            
                #Evaluate on dev set
                current_step = tf.train.global_step(sess, global_step)
                if current_step % args.checkpoint_steps == 0:
                    saver.save(sess, checkpoint_file, global_step=current_step)
                    print("Save model to %s" % checkpoint_file)
                if current_step % args.evaluate_steps == 0:
                    x_dev_batches = utils.batch_iter(x_dev, args.sequence_length, args.batch_size)
                    y_dev_batches = utils.batch_iter(y_dev, args.sequence_length, args.batch_size)
                    dev_losses = 0.0
                    i = 0
                    for x_dev_batch, y_dev_batch in zip(x_dev_batches, y_dev_batches):
                        dev_feed_dict = {rnn.input_data: x_dev_batch, 
                                         rnn.targets: y_dev_batch, 
                                         rnn.input_keep_prob: 1.0, 
                                         rnn.output_keep_prob: 1.0, 
                                         rnn.initial_state: initial_state}
                        step, loss = sess.run([global_step, rnn.loss], feed_dict=dev_feed_dict)
                        time_str = datetime.datetime.now().isoformat()
                        dev_losses += loss
                        i += 1
                    loss = dev_losses / i
                    print("Evaluate on dev set:")
                    print("{}: step {}, loss {:g}".format(time_str, step, loss))
                
                    if dev_loss > loss:
                        dev_loss = loss
                        saver.save(sess, dev_file, global_step=current_step)
                        print("Save better model to %s" % dev_file)