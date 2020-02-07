#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict2 -  Tensorflow
* Perform Machine Learning on Spectroscopy Data.
*
* Uses: Deep Neural Networks, TensorFlow, SVM, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''

import matplotlib
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Agg')

import numpy as np
import sys, os.path, getopt, glob, csv
import random, time, configparser, os
from os.path import exists, splitext
from os import rename
from datetime import datetime, date

from .slp_config import *

#********************************************************************************
''' TensorFlow '''
''' Basic Tensorflow '''
''' https://www.tensorflow.org/get_started/mnist/beginners'''
#********************************************************************************
''' Train basic TF training via TensorFlow- '''
#********************************************************************************
def trainTF(A, Cl, A_test, Cl_test, Root):
    print('==========================================================================\n')
    print('\033[1m Running Basic TensorFlow...\033[0m')

    import tensorflow as tf
    import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing
    
    if tfDef.logCheckpoint == True:
        tf.logging.set_verbosity(tf.logging.INFO)
    
    tfTrainedData = Root + '.tfmodel'
    print("\n  Training model saved in: ", tfTrainedData, "\n")

    #**********************************************
    ''' Initialize Estimator and training data '''
    #**********************************************
    print(' Initializing TensorFlow...')
    tf.reset_default_graph()

    totA = np.vstack((A, A_test))
    totCl = np.append(Cl, Cl_test)
    numTotClasses = np.unique(totCl).size
    
    le = preprocessing.LabelBinarizer()
    totCl2 = le.fit_transform(totCl) # this is original from DNNTF
    Cl2 = le.transform(Cl)     # this is original from DNNTF
    Cl2_test = le.transform(Cl_test)
    
    #validation_monitor = skflow.monitors.ValidationMonitor(input_fn=lambda: input_fn(A_test, Cl2_test),
    #                                                       eval_steps=1,
    #                                                       every_n_steps=dnntfDef.valMonitorSecs)

    #**********************************************
    ''' Construct TF model '''
    #**********************************************
    x,y,y_ = setupTFmodel(totA, totCl)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    
    if tfDef.decayLearnRate == True:
        print(' Using decaying learning rate, start at:',tfDef.learnRate, '\n')
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = tfDef.learnRate
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 100000, 0.96, staircase=True)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)
    else:
        print(' Using fix learning rate:', tfDef.learnRate, '\n')
        train_step = tf.train.GradientDescentOptimizer(tfDef.learnRate).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    
    if tfDef.enableTensorboard == True:
        writer = tf.summary.FileWriter(".", sess.graph)
        print('\n Saving graph. Accessible via tensorboard.  \n')

    saver = tf.train.Saver()
    accur = 0

    #**********************************************
    ''' Train '''
    #**********************************************
    try:
        if tfDef.alwaysRetrain == False:
            print(' Opening TF training model from:', tfTrainedData)
            saver.restore(sess, './' + tfTrainedData)
            print('\n Model restored.\n')
        else:
            raise ValueError(' Force TF model retraining.')
    except:
        init = tf.global_variables_initializer()
        sess.run(init)
        if os.path.isfile(tfTrainedData + '.meta') & tfDef.alwaysImprove == True:
            print('\n Improving TF model...')
            saver.restore(sess, './' + tfTrainedData)
        else:
            print('\n Rebuildind TF model...')

        print(' Performing training using subset (' +  str(tfDef.percentCrossValid*100) + '%)')

        summary = sess.run(train_step, feed_dict={x: A, y_: Cl2})
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_score = 100*accuracy.eval(feed_dict={x:A_test, y_:Cl2_test})

        save_path = saver.save(sess, "./"+tfTrainedData)
        print(' Model saved in file: %s\n' % save_path)
        
    if tfDef.enableTensorboard == True:
        writer.close()
    
    sess.close()

    print('\n  ================================')
    print('  \033[1mDNN-TF\033[0m - Accuracy')
    print('  ================================')
    print("\n  Accuracy: {:.2f}%".format(accuracy_score))
    #print("  Loss: {:.2f}".format(accuracy_score["loss"]))
    #print("  Global step: {:.2f}\n".format(accuracy_score["global_step"]))
    print('  ================================\n')

#**********************************************
''' Predict using basic Tensorflow '''
#**********************************************
def predTF(A, Cl, R, Root):
    print('==========================================================================\n')
    print('\033[1m Running Basic TensorFlow Prediction...\033[0m')
    
    import tensorflow as tf
    import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing
    
    if tfDef.logCheckpoint == True:
        tf.logging.set_verbosity(tf.logging.INFO)
    
    tfTrainedData = Root + '.tfmodel'

    x,y,y_ = setupTFmodel(A, Cl)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    print(' Opening TF training model from:', tfTrainedData)
    saver = tf.train.Saver()
    saver.restore(sess, './' + tfTrainedData)
    
    res1 = sess.run(y, feed_dict={x: R})
    res2 = sess.run(tf.argmax(y, 1), feed_dict={x: R})
    
    sess.close()
    
    rosterPred = np.where(res1[0]>tfDef.thresholdProbabilityTFPred)[0]
    print('\n  ==============================')
    print('  \033[1mTF\033[0m - Probability >',str(tfDef.thresholdProbabilityTFPred),'%')
    print('  ==============================')
    print('  Prediction\tProbability [%]')
    for i in range(rosterPred.shape[0]):
        print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.1f}'.format(res1[0][rosterPred][i])))
    print('  ==============================\n')
    
    print('\033[1m Predicted value (TF): ' + str(np.unique(Cl)[res2][0]) + ' (Probability: ' + str('{:.1f}'.format(res1[0][res2][0])) + '%)\n' + '\033[0m' )
    return np.unique(Cl)[res2][0], res1[0][res2][0]

#**********************************************
''' Setup Tensorflow Model'''
#**********************************************
def setupTFmodel(A, Cl):
    import tensorflow as tf
    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, [None, A.shape[1]])
    W = tf.Variable(tf.zeros([A.shape[1], np.unique(Cl).shape[0]]),name="W")
    b = tf.Variable(tf.zeros(np.unique(Cl).shape[0]),name="b")
    y_ = tf.placeholder(tf.float32, [None, np.unique(Cl).shape[0]])
    
    # The raw formulation of cross-entropy can be numerically unstable
    #y = tf.nn.softmax(tf.matmul(x, W) + b)
    #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), axis=[1]))
    
    # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
    # outputs of 'y', and then average across the batch.
    y = tf.matmul(x,W) + b
    
    return x, y, y_

