from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import csv
import time
import json
import argparse
import numpy as np

sys.path.append('../')

def loadData():
    '''
    example method for loading data
    :return:
    '''
    return None, None, None, None, None, None

import tensorflow as tf

from vanilla.cnn import MNISTcnn
#from tensorflow.examples.tutorials.mnist import input_data

def predict(sess, x, keep_prob, pred, Xtest, Ytest, output_file):
    feed_dict = {x:Xtest, keep_prob: 1.0}
    prediction = sess.run(pred, feed_dict=feed_dict)

    with open(output_file, "w") as file:
        writer = csv.writer(file, delimiter = ",")
        writer.writerow(["id","label"])
        for i in range(len(prediction)):
            writer.writerow([str(i), str(prediction[i])])

    print("Output prediction: {0}". format(output_file))


def train(args, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest):
    num_class = 10

    x = tf.placeholder(tf.float32, (None, 28*28))
    y = tf.placeholder(tf.float32, (None, num_class))
    model = MNISTcnn(x, y, args)

    optimizer = tf.train.AdamOptimizer(1e-5).minimize(model.loss)

    saver = tf.train.Saver(tf.trainable_variables())

    with tf.Session() as sess:
        print('Starting training')
        sess.run(tf.global_variables_initializer())
        if args.load_params:
            ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
            print('Restoring parameters from', ckpt_file)
            saver.restore(sess, ckpt_file)

        num_batches = Xtrain.shape[0] // args.batch_size
       
        validation = True
        val_num_batches = Xval.shape[0] // args.batch_size

        test_num_batches = Xtest.shape[0] // args.batch_size

        best_validate_accuracy = 0
        score = 0

        norms = []

        for epoch in range(args.epochs):
            begin = time.time()

            # train
            train_accuracies = []
            norm_sub = []
            for i in range(num_batches):

                batch_x = Xtrain[i*args.batch_size:(i+1)*args.batch_size,:]
                batch_y = Ytrain[i*args.batch_size:(i+1)*args.batch_size,:]

                _, acc, norm = sess.run([optimizer, model.accuracy, model.norm], feed_dict={x: batch_x, y: batch_y, model.keep_prob: 0.5, model.keep_prob2: min(1, 0.5*float(epoch)/10+0.5)})
                # if i%5!=4:
                #     print (acc, end='\t')
                # else:
                #     print (acc)
                train_accuracies.append(acc)
                norm_sub.append(norm)
            train_acc_mean = np.mean(train_accuracies)
            # print ()
            norm_sub_value = np.mean(norm_sub)
            norms.append(norm_sub_value)

            # compute loss over validation data
            if validation:
                val_accuracies = []
                for i in range(val_num_batches):
                    batch_x = Xval[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_y = Yval[i*args.batch_size:(i+1)*args.batch_size,:]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y, model.keep_prob: 1.0, model.keep_prob2: 1.0})
                    val_accuracies.append(acc)
                val_acc_mean = np.mean(val_accuracies)

                # log progress to console
                print("Epoch %d, time = %ds, train accuracy = %.4f, validation accuracy = %.4f, norm = %.4f" % (epoch, time.time()-begin, train_acc_mean, val_acc_mean, norm_sub_value))
            else:
                print("Epoch %d, time = %ds, train accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean))
            sys.stdout.flush()

            if val_acc_mean > best_validate_accuracy:
                best_validate_accuracy = val_acc_mean

                test_accuracies = []
                for i in range(test_num_batches):
                    batch_x = Xtest[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_y = Ytest[i*args.batch_size:(i+1)*args.batch_size,:]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y, model.keep_prob: 1.0})
                    test_accuracies.append(acc)
                score = np.mean(test_accuracies)

                print("Best Validated Model Prediction Accuracy = %.4f " % (score))

            if (epoch + 1) % 10 == 0:
                ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
                saver.save(sess, ckpt_file)

        ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
        saver.save(sess, ckpt_file)

        print("Best Validated Model Prediction Accuracy = %.4f " % (score))

        # return score

        # predict test data
        # predict(sess, x, model.keep_prob, model.pred, Xtest, Ytest, args.output)
        
        
        # origiinal test data from 'http://yann.lecun.com/exdb/mnist/'
        # """
        # acc = sess.run(model.accuracy, feed_dict={x: data.test.images, y: data.test.labels, model.keep_prob: 1.0})
        # print("test accuracy %g"%acc)
        # """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output",  type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=250, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    args = parser.parse_args()

    # pretty print args
    # print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':')))

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = loadData()
    tf.set_random_seed(100)
    np.random.seed(100)
    train(args, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)