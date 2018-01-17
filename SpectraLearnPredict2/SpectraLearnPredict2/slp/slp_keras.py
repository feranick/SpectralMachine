#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict2 -  Keras
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

#**********************************************
''' Format input data for Estimator '''
#**********************************************
def input_fn(A, Cl2):
    import tensorflow as tf
    x = tf.constant(A.astype(np.float32))
    y = tf.constant(Cl2)
    return x,y

#********************************************************************************
''' Keras '''
''' https://keras.io/getting-started/sequential-model-guide/#examples'''
#********************************************************************************
def trainKeras(A, Cl, A_test, Cl_test, Root):

    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    from keras.optimizers import SGD
    from keras import regularizers
    from sklearn import preprocessing
    from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
    import tensorflow as tf
    
    printInfoKeras()

    #**********************************************
    ''' Initialize Estimator and training data '''
    #**********************************************
    print(' Preprocessing data and classes for Keras\n')

    totA = np.vstack((A, A_test))
    totCl = np.append(Cl, Cl_test)
    
    numTotClasses = np.unique(totCl).size
    le = preprocessing.LabelEncoder()
    totCl2 = le.fit_transform(totCl)
    Cl2 = le.transform(Cl)
    Cl2_test = le.transform(Cl_test)

    totCl2 = keras.utils.to_categorical(totCl2, num_classes=np.unique(totCl).size)
    Cl2 = keras.utils.to_categorical(Cl2, num_classes=np.unique(Cl).size+1)
    Cl2_test = keras.utils.to_categorical(Cl2_test, num_classes=np.unique(Cl).size+1)
    
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    for numLayers in kerasDef.hidden_layers:
        model.add(Dense(numLayers,
                    activation=kerasDef.activation_function,
                    input_dim=A.shape[1],
                    kernel_regularizer=regularizers.l2(kerasDef.l2_reg_strength)))
        model.add(Dropout(kerasDef.dropout_perc))
    model.add(Dense(np.unique(Cl).size+1, activation='softmax'))

    sgd = SGD(lr=kerasDef.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
              optimizer=kerasDef.optimizer,
              metrics=['accuracy'])

    model.fit(A, Cl2,
          epochs=kerasDef.trainingSteps,
          batch_size=128)
    score = model.evaluate(A_test, Cl2_test, batch_size=128)

    return model, le
def printInfoKeras():
    print('==========================================================================\n')
    print('\033[1m Running Deep Neural Networks: Keras...\033[0m')
    print('  Optimizer:',kerasDef.optimizer,
                '\n  Hidden layers:', kerasDef.hidden_layers,
                '\n  Activation function:',kerasDef.activation_function,
                '\n  L2:',kerasDef.l2_reg_strength,
                '\n  Dropout:', kerasDef.dropout_perc,
                '\n  Learning rate:', kerasDef.learning_rate,
                '\n')

#********************************************************************************
''' Predict using tf.estimator.DNNClassifier model via TensorFlow '''
#********************************************************************************
def predKeras(clf, le, R, Cl):
    import tensorflow as tf
    import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": R},
      num_epochs=1,
      shuffle=False)
      
    predictions = list(clf.predict(input_fn=predict_input_fn))
    pred_class = [p["class_ids"] for p in predictions][0][0]
    predValue = le.inverse_transform(pred_class)
    prob = [p["probabilities"] for p in predictions][0]
    predProb = round(100*prob[pred_class],2)
    
    rosterPred = np.where(prob>dnntfDef.thresholdProbabilityPred/100)[0]
    
    print('\n  ==================================')
    print('  \033[1mtf.DNN-TF\033[0m - Probability >',str(dnntfDef.thresholdProbabilityPred),'%')
    print('  ==================================')
    print('  Prediction\tProbability [%]')
    for i in range(rosterPred.shape[0]):
        print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.4f}'.format(100*prob[rosterPred][i])))
    print('  ==================================')
    
    print('\033[1m' + '\n Predicted value (tf.DNNClassifier) = ' + predValue +
          '  (probability = ' + str(predProb) + '%)\033[0m\n')

    return predValue, predProb


