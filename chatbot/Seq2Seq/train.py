# coding: utf-8

import os
import datetime
import argparse

import numpy as np
import tensorflow as tf

from model import Seq2SeqModel
import utils


def hparams():
    parser = argparse.ArgumentParser()
    
    # Data loading params
    parser.add_argument("--text_file", type=str, default="data/corpus.txt", help="Data source for the corpus")
    parser.add_argument("--vocab_file", type=str, default="data/vocab.txt", help="Vocabulary")
    parser.add_argument("--word2vec_file", type=str, default=None, help="Chinese wiki word2vec")
    
    # Model Parameters
    parser.add_argument("--vocab_size", type=int, default=50001, help="Vocabulary size")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimensionality of the words embedding")
    parser.add_argument("--rnn_size", type=int, default=256, help="Hidden units of rnn layer ")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of rnn layer")
    
    # Training Parameters
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--num_epochs", type=int, default=5000, help="Number of iterations")
    parser.add_argument("--input_keep_prob", type=float, default=0.5, help="Dropout rate of embedding layer")
    parser.add_argument('--output_keep_prob', type=float, default=0.5, help="Dropout rate of rnn layer")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, help="Clip global grad norm")
    parser.add_argument("--checkpoint_steps", type=int, default=50, help="Save model after this many steps")

    args, _ = parser.parse_known_args()
    return args


def train():
    args = hparams()
    
    # Load data
    print("Loading data...")
    encoder_inputs, decoder_inputs, decoder_outputs = utils.load_data(args.text_file)
    
    print("Example of query : %s" %encoder_inputs[10])
    print("Example of answer : %s" %decoder_outputs[10])

    # Build vocabulary
    # text = encoder_inputs + decoder_inputs
    # utils.get_vocab(args.vocab_file, text)
    vocab_dict = utils.load_vocab(args.vocab_file)
    
    print("Example of query to ids: \n %s" %utils.encoder_transform(encoder_inputs[:100], vocab_dict)[10])
    print("Example of answer to ids: \n %s" %utils.decoder_transform(decoder_outputs[:100], vocab_dict)[10])
    
    # Training
    sess = tf.Session()
    with sess.as_default():
        model = Seq2SeqModel(vocab_size=args.vocab_size, 
                             embedding_dim=args.embedding_dim, 
                             rnn_size=args.rnn_size, 
                             num_layers=args.num_layers, 
                             batch_size=args.batch_size, 
                             word2vec_path=args.word2vec_file, 
                             vocab_path=args.vocab_file, 
                             training=True)
        
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(args.learning_rate)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), args.max_grad_norm)
        train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        
        # Save model params
        checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_file = os.path.join(checkpoint_dir, "model")
        saver = tf.train.Saver(tf.global_variables())
        
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        
        # Generate batches
        batches = utils.batch_iter(list(zip(encoder_inputs, decoder_inputs, decoder_outputs)), 
                                   args.batch_size, args.num_epochs, False)
        
        initial_state = sess.run(model.initial_state)
        # Training loop...
        for batch in batches:
            x_input, y_input, y_output = zip(*batch)
            x_input = tuple(utils.encoder_transform(list(x_input), vocab_dict))
            y_input = tuple(utils.decoder_transform(list(y_input), vocab_dict))
            y_output = tuple(utils.decoder_transform(list(y_output), vocab_dict))
            feed_dict = {model.encoder_inputs: x_input, 
                         model.decoder_inputs: y_input, 
                         model.decoder_outputs: y_output, 
                         model.initial_state: initial_state, 
                         model.input_keep_prob: args.input_keep_prob,
                         model.output_keep_prob: args.output_keep_prob}
            _, step, loss = sess.run([train_op, global_step, model.loss], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))
            
            # Evaluate on dev set
            current_step = tf.train.global_step(sess, global_step)
            if current_step % args.checkpoint_steps == 0:
                saver.save(sess, checkpoint_file, global_step=current_step)
                print("Save model to %s" % checkpoint_file)


def sub_train():
    args = hparams()
    
    # Load data
    print("Loading data...")
    encoder_inputs, decoder_inputs, decoder_outputs = utils.load_data(args.text_file)
    
    print("Example of query : %s" %encoder_inputs[10])
    print("Example of answer : %s" %decoder_outputs[10])

    # Build vocabulary
    # text = encoder_inputs + decoder_inputs
    # utils.get_vocab(args.vocab_file, text)
    vocab_dict = utils.load_vocab(args.vocab_file)
    
    print("Example of query to ids: \n %s" %utils.encoder_transform(encoder_inputs[:100], vocab_dict)[10])
    print("Example of answer to ids: \n %s" %utils.decoder_transform(decoder_outputs[:100], vocab_dict)[10])
    
    # Save model params
    checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir, "checkpoints"))
    checkpoint_file = os.path.join(checkpoint_dir, "model")
    
    # Training
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
                                 training=True)
        
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(args.learning_rate)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), args.max_grad_norm)
            train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        
        
        
            # Load model variables
            saver = tf.train.Saver(tf.global_variables())
            last_checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
            saver.restore(sess, last_checkpoint_file)
        
            # Generate batches
            batches = utils.batch_iter(list(zip(encoder_inputs, decoder_inputs, decoder_outputs)), 
                                       args.batch_size, args.num_epochs, False)
        
            # Training loop...
            initial_state = sess.run(model.initial_state)
            for batch in batches:
                x_input, y_input, y_output = zip(*batch)
                x_input = tuple(utils.encoder_transform(list(x_input), vocab_dict))
                y_input = tuple(utils.decoder_transform(list(y_input), vocab_dict))
                y_output = tuple(utils.decoder_transform(list(y_output), vocab_dict))
                feed_dict = {model.encoder_inputs: x_input, 
                             model.decoder_inputs: y_input, 
                             model.decoder_outputs: y_output, 
                             model.initial_state: initial_state, 
                             model.input_keep_prob: args.input_keep_prob,
                             model.output_keep_prob: args.output_keep_prob}
                _, step, loss = sess.run([train_op, global_step, model.loss], feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}".format(time_str, step, loss))
            
                # Evaluate on dev set
                current_step = tf.train.global_step(sess, global_step)
                if current_step % args.checkpoint_steps == 0:
                    saver.save(sess, checkpoint_file, global_step=current_step)
                    print("Save model to %s" % checkpoint_file)
