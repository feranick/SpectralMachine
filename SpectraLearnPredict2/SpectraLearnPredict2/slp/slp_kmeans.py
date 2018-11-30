#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict2 - kmeans
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


#********************
''' Run K-Means '''
#********************
def runKMmain(A, Cl, En, R, Aorig, Rorig):
    from sklearn.cluster import KMeans
    print('==========================================================================\n')
    print(' Running K-Means...')
    print(' Number of unique identifiers in training data: ' + str(np.unique(Cl).shape[0]))
    if kmDef.customNumKMComp == False:
        numKMcomp = np.unique(Cl).shape[0]
    else:
        numKMcomp = kmDef.numKMcomponents
    kmeans = KMeans(n_clusters=numKMcomp, random_state=0).fit(A)
    prediction = kmeans.predict(R)[0]

    print('\n  ==============================')
    print('  \033[1mK-Means\033[0m - Prediction')
    print('  ==============================')
    print('  Class\t| Value')
    for j in range(0,kmeans.labels_.shape[0]):
        if kmeans.labels_[j] == prediction:
            print("  {0:d}\t| {1:.2f}".format(prediction,Cl[j]))
    print('  ==============================\n')

    if kmDef.plotKM == True:
        import matplotlib.pyplot as plt
        for j in range(0,kmeans.labels_.shape[0]):
            if kmeans.labels_[j] == kmeans.predict(R)[0]:
                plt.plot(En, Aorig[j,:])
        plt.plot(En, Rorig[0,:], linewidth = 2, label='Predict')
        plt.title('K-Means')
        plt.xlabel('Raman shift [1/cm]')
        plt.ylabel('Intensity')
        plt.legend()
        plt.show()
    return kmeans.predict(R)[0]

#**********************************************
''' K-Means - Maps'''
#**********************************************
def KmMap(mapFile, numKMcomp):
    ''' Open prediction map '''
    X, Y, R, Rx = readPredMap(mapFile)
    type = 0
    i = 0;
    R, Rx, Rorig = preProcessNormMap(R, Rx, type)

    from sklearn.cluster import KMeans
    print(' Running K-Means...')
    print(' Number of classes: ' + str(numKMcomp))
    kmeans = KMeans(n_clusters=kmDef.numKMcomponents, random_state=0).fit(R)
    kmPred = np.empty([R.shape[0]])

    for i in range(0, R.shape[0]):
        kmPred[i] = kmeans.predict(R[i,:].reshape(1,-1))[0]
        saveMap(mapFile, 'KM', 'Class', int(kmPred[i]), X[i], Y[i], True)
        if kmPred[i] in kmeans.labels_:
            if os.path.isfile(saveMapName(mapFile, 'KM', 'Class_'+ str(int(kmPred[i]))+'-'+str(np.unique(kmeans.labels_).shape[0]), False)) == False:
                saveMap(mapFile, 'KM', 'Class_'+ str(int(kmPred[i])) + '-'+str(np.unique(kmeans.labels_).shape[0]) , '\t'.join(map(str, Rx)), ' ', ' ', False)
            saveMap(mapFile, 'KM', 'Class_'+ str(int(kmPred[i])) + '-'+str(np.unique(kmeans.labels_).shape[0]) , '\t'.join(map(str, R[1,:])), X[i], Y[i], False)

    if kmDef.plotKM == True:
        plotMaps(X, Y, kmPred, 'K-Means')

