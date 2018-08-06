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
import heapq

sys.path.append('../')


import tensorflow as tf

from CF.cnn import MNISTcnn
#from tensorflow.examples.tutorials.mnist import input_data

def loadData():
    '''
    example method for loading data
    '''
    return None, None, None, None, None, None

def loadConfoundingFactorLabels():
    '''
    example method for loading data
    '''
    return None, None, None


def find(value_list, arr):
    
    max_row = []
    max_rank = []
    
    for li in range(0, len(value_list)):
        
        value = value_list[li] 
        for i in range(0, arr.shape[0]):
            for j in range(0, arr.shape[1]):
                if arr[i][j] == value:
                    max_row.append(i)
                    max_rank.append(j)
    
    return max_row, max_rank

def find_index(three_dim_matrix):
    # three_dim_matrix : batch number, 1024,7
    num = len(three_dim_matrix)
    abs_matrix = np.zeros([1024, 10])
    for i in range(num - 1):
        abs_matrix = abs_matrix + abs(three_dim_matrix[i+1] - three_dim_matrix[i])      
    
    max_list = heapq.nlargest(2000, abs_matrix.flatten())
    row, rank = find(max_list, abs_matrix)
    
    return row, rank
        


def predict(sess, x, keep_prob, pred, Xtest, Ytest, output_file):
    feed_dict = {x:Xtest, keep_prob: 1.0}
    prediction = sess.run(pred, feed_dict=feed_dict)

    with open(output_file, "w") as file:
        writer = csv.writer(file, delimiter = ",")
        writer.writerow(["id","label"])
        for i in range(len(prediction)):
            writer.writerow([str(i), str(prediction[i])])

    print("Output prediction: {0}". format(output_file))


def train(step, args, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest):
    with tf.variable_scope('model',reuse=tf.AUTO_REUSE ) as scope:
        num_class = 10

        x = tf.placeholder(tf.float32, (None, 28*28))
        y = tf.placeholder(tf.float32, (None, num_class))
        model = MNISTcnn(args.ckpt_dir, step, x, y, args, Hex_flag=False)

        optimizer = tf.train.AdamOptimizer(1e-5).minimize(model.loss)
        tf.get_variable_scope().reuse_variables()

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
            
            
            for epoch in range(args.epochs):
                begin = time.time()
                changing_record = []
                
                if step != 2:
                    
                    # train
                    train_accuracies = []
                    for i in range(num_batches):
            
                        batch_x = Xtrain[i*args.batch_size:(i+1)*args.batch_size,:]
                        batch_y = Ytrain[i*args.batch_size:(i+1)*args.batch_size,:]
            
                        _, acc = sess.run([optimizer,model.accuracy], feed_dict={x: batch_x, y: batch_y, model.keep_prob: 0.5})
                        train_accuracies.append(acc)
                        
                        #step 1
                        if step == 0:
                            weight_matrix = sess.run([model.W_fc2], feed_dict={x: batch_x, y: batch_y, model.keep_prob: 1})
                            weight_matrix=np.array(weight_matrix).reshape(1024,10)
                            changing_record.append(weight_matrix)
                            
                            
                    #step 1
                    if step == 0:      
                        global top_rank,top_row
                        top_row, top_rank = find_index(changing_record) 
                    
                    train_acc_mean = np.mean(train_accuracies)
                    # print ()
            
                    # compute loss over validation data
                    if validation:
                        val_accuracies = []
                        for i in range(val_num_batches):
                            batch_x = Xval[i*args.batch_size:(i+1)*args.batch_size,:]
                            batch_y = Yval[i*args.batch_size:(i+1)*args.batch_size,:]
                            acc = sess.run(model.accuracy, feed_dict={x: batch_x, y: batch_y, model.keep_prob: 1.0})
                            val_accuracies.append(acc)
                        val_acc_mean = np.mean(val_accuracies)
            
                        # log progress to console
                        print("Epoch %d, time = %ds, train accuracy = %.4f, validation accuracy = %.4f" % (epoch, time.time()-begin, train_acc_mean, val_acc_mean))
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
                        
                        
                        if step == 1:
                            
                            w1, b1, w2, b2, wf1, bf1, modify_weight_matrix, bf2 = sess.run([model.W_conv1, model.b_conv1, model.W_conv2, model.b_conv2, model.W_fc1, model.b_fc1, model.W_fc2, model.b_fc2], feed_dict={x: batch_x, y: batch_y, model.keep_prob: 1})
                            for c in range(2000):
                                modify_weight_matrix[top_row[c]][top_rank[c]] = 0
                            
                            bin_file_path = os.path.join(args.ckpt_dir, 'w1.bin')
                            w1.tofile(bin_file_path)   

                            bin_file_path = os.path.join(args.ckpt_dir, 'b1.bin')
                            b1.tofile(bin_file_path)

                            bin_file_path = os.path.join(args.ckpt_dir, 'w2.bin')
                            w2.tofile(bin_file_path)

                            bin_file_path = os.path.join(args.ckpt_dir, 'b2.bin')
                            b2.tofile(bin_file_path)

                            bin_file_path = os.path.join(args.ckpt_dir, 'wf1.bin')
                            wf1.tofile(bin_file_path)

                            bin_file_path = os.path.join(args.ckpt_dir, 'bf1.bin')
                            bf1.tofile(bin_file_path)

                            bin_file_path = os.path.join(args.ckpt_dir, 'modify_w2.bin')
                            modify_weight_matrix.tofile(bin_file_path)

                            bin_file_path = os.path.join(args.ckpt_dir, 'bf2.bin')
                            bf2.tofile(bin_file_path)  
                            
                    if (epoch + 1) % 10 == 0:
                        ckpt_file = os.path.join(args.ckpt_dir, 'mnist_model.ckpt')
                        saver.save(sess, ckpt_file)
                    scope.reuse_variables()
                else:
                    
                    again_test_accuracies = []
                        
                    for again_i in range(test_num_batches):
                        again_batch_x = Xtest[again_i*args.batch_size:(again_i+1)*args.batch_size,:]
                        again_batch_y = Ytest[again_i*args.batch_size:(again_i+1)*args.batch_size,:]
                        again_acc = sess.run(model.accuracy, feed_dict={x: again_batch_x, y: again_batch_y, model.keep_prob: 1.0})
                        again_test_accuracies.append(again_acc)
                    again_score = np.mean(again_test_accuracies)
            
                    print("again: Best Validated Model Prediction Accuracy = %.4f " % (again_score))

                    
                    
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
    parser.add_argument('-e', '--epochs', type=int, default=300, help='How many epochs to run in total?')
    parser.add_argument('-b', '--batch_size', type=int, default=128, help='Batch size during training per GPU')
    args = parser.parse_args()

    tf.set_random_seed(100)
    np.random.seed(100)
  
    # pretty print args
    print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) 

    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = loadData()
    control_ytrain,control_yval,control_ytest= loadConfoundingFactorLabels()
    
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

  
    global top_row 
    global top_rank 
    
    #step 1->First train: train the model with confounder-related data to find the confounding dimension(step=0)
    #step 2->Second train: train the model with normal dataset(step=1)
    #step 3->Third Test: test again with the model after using the CF method(step=2)
    
    # Step0    
    train(0,args,Xtrain,control_ytrain,Xval,control_yval,Xtest,control_ytest)

    # step1
    train(1,args,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest)

    # step2
    train(2,args,Xtrain,Ytrain,Xval,Yval,Xtest,Ytest)
    
    """  
    for step in range(3):
        #tf.reset_default_graph() 
        train(step, args, Xtrain, Ytrain, Xval, Yval, Xtest, Ytest)
    """
