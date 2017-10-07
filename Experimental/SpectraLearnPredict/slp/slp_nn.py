#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict - SKlearn Neural Networks
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
''' MultiLayer Perceptron - SKlearn '''
''' http://scikit-learn.org/stable/modules/neural_networks_supervised.html'''
#********************************************************************************
''' Train Neural Network - sklearn '''
#********************************************************************************
def trainNN(A, Cl, A_test, Cl_test, Root):
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.externals import joblib
    
    if nnDef.MLPRegressor is False:
        Root+"/DNN-TF_"
        nnTrainedData = Root + '.nnModelC.pkl'
    else:
        nnTrainedData = Root + '.nnModelR.pkl'

    print('==========================================================================\n')
    print('\033[1m Running Neural Network: multi-layer perceptron (MLP)\033[0m')
    print('  Hidden layers with neuron count:', nnDef.hidden_layers)
    print('  Optimizer:',nnDef.optimizer,', Activation Fn:',nnDef.activation_function,
          ', L2 reg. strength: ',nnDef.l2_reg_strength)

    try:
        if nnDef.alwaysRetrain == False:
            with open(nnTrainedData):
                print('  Opening NN training model...\n')
                clf = joblib.load(nnTrainedData)
        else:
            raise ValueError('  Force NN retraining.')
    except:
        #**********************************************
        ''' Retrain training data if not available'''
        #**********************************************
        if nnDef.MLPRegressor is False:
            print('  Retraining NN model using MLP Classifier...')
            clf = MLPClassifier(solver=nnDef.optimizer, alpha=nnDef.l2_reg_strength,
                                activation = nnDef.activation_function,
                                hidden_layer_sizes=nnDef.hidden_layers, random_state=1)
        else:
            print('  Retraining NN model using MLP Regressor...')
            clf = MLPRegressor(solver=nnDef.optimizer, alpha=nnDef.l2_reg_strength,
                               hidden_layer_sizes=nnDef.hidden_layers, random_state=1)
            Cl = np.array(Cl,dtype=float)

        clf.fit(A, Cl)
        print("  Training on the full training dataset\n")
        accur = clf.score(A_test,Cl_test)

        if nnDef.MLPRegressor is False:
            print('  Accuracy: ',100*accur,'%\n  Loss: {:.5f}'.format(clf.loss_),'\n')
        else:
            print('  Coefficient of determination R^2: ',accur,
                  '\n  Loss: {:.5f}'.format(clf.loss_),'\n')

        joblib.dump(clf, nnTrainedData)

    return clf

#********************************************************************************
''' Evaluate Neural Network - sklearn '''
#********************************************************************************
def predNN(clf, A, Cl, R):
    if nnDef.MLPRegressor is False:
        prob = clf.predict_proba(R)[0].tolist()
        rosterPred = np.where(clf.predict_proba(R)[0]>nnDef.thresholdProbabilityPred/100)[0]
        print('\n  ==============================')
        print('  \033[1mNN\033[0m - Probability >',str(nnDef.thresholdProbabilityPred),'%')
        print('  ==============================')
        print('  Prediction\tProbability [%]')
        for i in range(rosterPred.shape[0]):
            print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.4f}'.format(100*clf.predict_proba(R)[0][rosterPred][i])))
        print('  ==============================')
        
        predValue = clf.predict(R)[0]
        predProb = round(100*max(prob),4)
        print('\033[1m' + '\n Predicted classifier value (Deep Neural Networks - sklearn) = ' + str(predValue) +
              '  (probability = ' + str(predProb) + '%)\033[0m\n')
    else:
        Cl = np.array(Cl,dtype=float)
        predValue = clf.predict(R)[0]
        predProb = clf.score(A,Cl)
        print('\033[1m' + '\n Predicted regressor value (Deep Neural Networks - sklearn) = ' + str('{:.3f}'.format(predValue)) +
              '  (R^2 = ' + str('{:.5f}'.format(predProb)) + ')\033[0m\n')
    
    #**************************************
    ''' Neural Networks Classification Report '''
    #**************************************
    if nnDef.nnClassReport == True:
        print(' Neural Networks Classification Report\n')
        runClassReport(clf, A, Cl)

    #*************************
    ''' Plot probabilities '''
    #*************************
    if plotDef.showProbPlot == True:
        if nnDef.MLPRegressor is False:
            plotProb(clf, R)

    return predValue, predProb

