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
    from keras.models import load_model
    from keras.callbacks import TensorBoard
    from sklearn import preprocessing
    from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
    import tensorflow as tf
    
    tb_directory = "keras_" + str(len(kerasDef.hidden_layers))+"HL_"+str(kerasDef.hidden_layers[0])
    model_directory = "."
    model_name = model_directory+"/keras_"+str(len(kerasDef.hidden_layers))+"HL_"+str(kerasDef.hidden_layers[0])+".hd5"
    
    if kerasDef.alwaysRetrain == False:
        print("\n  Training model saved in: ", model_name, "\n")
    else:
        kerasDef.alwaysImprove = False
        print("\n  Training model not saved\n")
    
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
    
    
    if kerasDef.alwaysImprove == True or os.path.exists(model_name) is False:
        model = Sequential()
        # Dense(64) is a fully-connected layer with 64 hidden units.
        # in the first layer, you must specify the expected input data shape:
        # here, 20-dimensional vectors.
        for numLayers in kerasDef.hidden_layers:
            model.add(Dense(numLayers,
                    activation = kerasDef.activation_function,
                    input_dim=A.shape[1],
                    kernel_regularizer=regularizers.l2(kerasDef.l2_reg_strength)))
            model.add(Dropout(kerasDef.dropout_perc))
        model.add(Dense(np.unique(Cl).size+1, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
              optimizer=kerasDef.optimizer,
              metrics=['accuracy'])

        tbLog = TensorBoard(log_dir=tb_directory, histogram_freq=0, batch_size=32,
                write_graph=True, write_grads=False, write_images=False,
                embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
        tbLogs = [tbLog]
        model.fit(A, Cl2,
            epochs=kerasDef.trainingSteps,
            batch_size=128,
            callbacks = tbLogs)

        score = model.evaluate(A_test, Cl2_test, batch_size=128)
        model.save(model_name)

        if kerasDef.plotModel == True:
            from keras.utils import plot_model
            plot_model(model, to_file=model_directory+'/model.png')
            

        print('\n  ==================================')
        print('  \033[1mKeras\033[0m - Accuracy')
        print('  ==================================')
        print("\n  Accuracy: {:.2f}%".format(100*score[1]))
        print("  Loss: {:.2f}".format(score[0]))
        print("  Global step: {:.2f}\n".format(kerasDef.trainingSteps))
        print('  ==================================\n')
        
    else:
        print(" Retreaving training model from: ", model_name,"\n")
        model = load_model(model_name)

    return model, le

def printInfoKeras():
    print('==========================================================================\n')
    print('\033[1m Running Deep Neural Networks: Keras...\033[0m')
    print('  Optimizer:',kerasDef.optimizer_tag,
                '\n  Hidden layers:', kerasDef.hidden_layers,
                '\n  Activation function:',kerasDef.activation_function,
                '\n  L2:',kerasDef.l2_reg_strength,
                '\n  Dropout:', kerasDef.dropout_perc,
                '\n  Learning rate:', kerasDef.learning_rate,
                '\n  Learning decay rate:', kerasDef.learning_decay_rate,
                '\n')

#********************************************************************************
''' Predict using Keras model '''
#********************************************************************************
def predKeras(model, le, R, Cl):
    import keras
    from sklearn import preprocessing

    predictions = model.predict(R, verbose=1)
    pred_class = np.argmax(predictions)
    if pred_class.size >0:
        predValue = le.inverse_transform(pred_class)
    else:
        predValue = 0

    predProb = round(100*predictions[0][pred_class],2)
    rosterPred = np.where(predictions[0]>dnntfDef.thresholdProbabilityPred)[0]
    
    print('\n  ==================================')
    print('  \033[1mKeras\033[0m - Probability >',str(kerasDef.thresholdProbabilityPred),'%')
    print('  ==================================')
    print('  Prediction\tProbability [%]')
    for i in range(rosterPred.shape[0]):
        print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',
            str('{:.4f}'.format(100*predictions[0][rosterPred][i])))
    print('  ==================================')
    
    print('\033[1m' + '\n Predicted value (tf.DNNClassifier) = ' + predValue +
          '  (probability = ' + str(predProb) + '%)\033[0m\n')

    return predValue, predProb



