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
    X: data
    Y: labels
    train: training data
    val: validation data
    test: testing data
    '''
    Xtrain = None
    Ytrain = None
    Xval = None
    Yval = None
    Xtest = None
    Ytest = None
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest

def loadCFLabels():
    '''
    corresponding labels of confounding factors
    The method does not really need Zval or Ztest
    '''
    Ztrain = None
    Zval = None
    Ztest = None
    return Ztrain, Zval, Ztest

import tensorflow as tf

from CF.cnn import MNISTcnn
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


def train(args, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, Ztrain):
    num_class = 10

    x = tf.placeholder(tf.float32, (None, 28*28))
    y = tf.placeholder(tf.float32, (None, num_class))
    model = MNISTcnn(x, y, args)

    optimizer = tf.train.AdamOptimizer(1e-5).minimize(model.loss)

    saver = tf.train.Saver(tf.trainable_variables())

    weights = {}

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

        # Phase One
        for epoch in range(args.epochs):
            begin = time.time()

            # train
            train_accuracies = []
            for i in range(num_batches):

                batch_x = Xtrain[i*args.batch_size:(i+1)*args.batch_size,:]
                batch_y = Ytrain[i*args.batch_size:(i+1)*args.batch_size,:]

                _, acc = sess.run([optimizer, model.accuracy], feed_dict={x: batch_x, y: batch_y})
                train_accuracies.append(acc)
            train_acc_mean = np.mean(train_accuracies)

            # compute loss over validation data
            if validation:
                val_accuracies = []
                for i in range(val_num_batches):
                    batch_x = Xval[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_y = Yval[i*args.batch_size:(i+1)*args.batch_size,:]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y})
                    val_accuracies.append(acc)
                val_acc_mean = np.mean(val_accuracies)

                # log progress to console
                print("Epoch %d, time = %ds, train accuracy = %.4f, validation accuracy = %.4f" % (
                epoch, time.time() - begin, train_acc_mean, val_acc_mean))
            else:
                print("Epoch %d, time = %ds, train accuracy = %.4f" % (epoch, time.time() - begin, train_acc_mean))
            sys.stdout.flush()

            if val_acc_mean > best_validate_accuracy:
                best_validate_accuracy = val_acc_mean

                test_accuracies = []
                for i in range(test_num_batches):
                    batch_x = Xtest[i*args.batch_size:(i+1)*args.batch_size,:]
                    batch_y = Ytest[i*args.batch_size:(i+1)*args.batch_size,:]
                    acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y})
                    test_accuracies.append(acc)
                score = np.mean(test_accuracies)

                print("Best Validated Model Prediction Accuracy = %.4f " % (score))

            if (epoch + 1) % 10 == 0:
                ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
                saver.save(sess, ckpt_file)

        for v in tf.trainable_variables():
            weights[v.name] = v.eval()

        # Phase Two
        weight_pre = weights[args.layer]
        changes = np.zeros_like(weights)
        for epoch in range(args.epochs_cf):
            begin = time.time()

            # train
            for i in range(num_batches):

                batch_x = Xtrain[i*args.batch_size:(i+1)*args.batch_size,:]
                batch_z = Ztrain[i*args.batch_size:(i+1)*args.batch_size,:]

                _, acc, weight = sess.run([optimizer, model.accuracy, model.layer], feed_dict={x: batch_x, y: batch_z})
                changes += np.abs(weight - weight_pre)/np.max(np.abs(weight - weight_pre))
                weight_pre = weight
        changes = changes/(args.epochs_cf*num_batches)


        # Phase Three
        weights[args.layer][changes>args.threshold] = 0
        model.setWeights(sess, weights)

        ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
        saver.save(sess, ckpt_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--ckpt_dir', type=str, default='ckpts/', help='Directory for parameter checkpoints')
    parser.add_argument('-l', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
    parser.add_argument("-o", "--output",  type=str, default='prediction.csv', help='Prediction filepath')
    parser.add_argument('-e', '--epochs', type=int, default=250, help='How many epochs to run in first phase?')
    parser.add_argument('-p', '--epochs_cf', type=int, default=50, help='How many epochs to run in second phase?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    parser.add_argument('-t', '--threshold', type=int, default=0.75, help='threshold of updating the weights')
    parser.add_argument('-n', '--layer', type=str, default='fc1_weights', help='the layer we are interested in adjust')
    parser.add_argument('-s', '--seed', type=int, default=0, help='random seed for generating data')
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = loadData()
    Ztrain, Zval, Ztest = loadCFLabels()
    tf.set_random_seed(100)
    np.random.seed(100)
    train(args, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, Ztrain)
