#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict2 - Preprocess data
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
import random, time, configparser, os, h5py
from os.path import exists, splitext
from os import rename
from datetime import datetime, date

from .slp_config import *

#************************************
''' Read Learning file '''
#************************************
def readLearnFile(learnFile):
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        elif os.path.splitext(learnFile)[1] == ".h5":
            with h5py.File(learnFile, 'r') as hf:
                M = hf["M"][:]
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Learning file not found \n' + '\033[0m')
        return

    '''
    # Obsolete
    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(M,np.s_[0:1],0)
    Cl = ['{:.2f}'.format(x) for x in M[:,0]]
    A = np.delete(M,np.s_[0:1],1)
    '''

    En = M[0,1:]
    A = M[1:,1:]
    Cl = M[1:,0]
    
    Atemp = A[:,range(len(preprocDef.enSel))]

    if preprocDef.cherryPickEnPoint == True and preprocDef.enRestrictRegion == False:
        enPoints = list(preprocDef.enSel)
        enRange = list(preprocDef.enSel)

        for i in range(0, len(preprocDef.enSel)):
            enRange[i] = np.where((En<float(preprocDef.enSel[i]+preprocDef.enSelDelta[i])) & (En>float(preprocDef.enSel[i]-preprocDef.enSelDelta[i])))[0].tolist()

            for j in range(0, A.shape[0]):
                Atemp[j,i] = A[j,A[j,enRange[i]].tolist().index(max(A[j, enRange[i]].tolist()))+enRange[i][0]]

            enPoints[i] = int(np.average(enRange[i]))
        A = Atemp
        En = En[enPoints]

        if type == 0:
            print( ' Cheery picking points in the spectra\n')

    # Find index corresponding to energy value to be used for Y normalization
    if preprocDef.fullYnorm == True:
        YnormXind = np.where(En>0)[0].tolist()
    else:
        YnormXind_temp = np.where((En<float(preprocDef.YnormX+preprocDef.YnormXdelta)) & (En>float(preprocDef.YnormX-preprocDef.YnormXdelta)))[0].tolist()
        if YnormXind_temp == []:
            print( ' Renormalization region out of requested range. Normalizing over full range...\n')
            YnormXind = np.where(En>0)[0].tolist()
        else:
            YnormXind = YnormXind_temp

    print(' Number of spectra = ' + str(A.shape[0]))
    print(' Number of points in each spectra = ' + str(A.shape[1]))
    print(' Number of unique classes = ' + str(len(np.unique(Cl))) + '\n')
    return En, Cl, A, YnormXind

#**********************************************
''' Open prediction file '''
#**********************************************
def readPredFile(sampleFile):
    try:
        with open(sampleFile, 'r') as f:
            print(' Opening sample data for prediction...')
            Rtot = np.loadtxt(f, unpack =True)
    except:
        print('\033[1m' + '\n Sample data file not found \n ' + '\033[0m')
        return

    R=Rtot[1,:]
    Rx=Rtot[0,:]

    if preprocDef.cherryPickEnPoint == True and preprocDef.enRestrictRegion == False:
        Rtemp = R[range(len(preprocDef.enSel))]
        enPoints = list(preprocDef.enSel)
        enRange = list(preprocDef.enSel)
        for i in range(0, len(preprocDef.enSel)):
            enRange[i] = np.where((Rx<float(preprocDef.enSel[i]+preprocDef.enSelDelta[i])) & (Rx>float(preprocDef.enSel[i]-preprocDef.enSelDelta[i])))[0].tolist()
            Rtemp[i] = R[R[enRange[i]].tolist().index(max(R[enRange[i]].tolist()))+enRange[i][0]]

            enPoints[i] = int(np.average(enRange[i]))
        R = Rtemp
        Rx = Rx[enPoints]

    return R, Rx

#**********************************************************************************
''' Preprocess Learning data '''
#**********************************************************************************
def preProcessNormLearningData(A, En, Cl, YnormXind, type):
    print(' Processing dataset ... ')
    #**********************************************************************************
    ''' Reformat x-axis in case it does not match that of the training data '''
    #**********************************************************************************
    if preprocDef.scrambleNoiseFlag == True:
        print(' Adding random noise to training set \n')
        scrambleNoise(A, preprocDef.scrambleNoiseOffset)
    Aorig = np.copy(A)

    #**********************************************
    ''' Normalize/preprocess if flags are set '''
    #**********************************************
    if preprocDef.Ynorm == True:
        if type == 0:
            if preprocDef.fullYnorm == False:
                print('  Normalizing spectral intensity to: ' + str(preprocDef.YnormTo) + '; En = [' + str(preprocDef.YnormX-preprocDef.YnormXdelta) + ', ' + str(preprocDef.YnormX+preprocDef.YnormXdelta) + ']')
            else:
                print('  Normalizing spectral intensity to: ' + str(preprocDef.YnormTo) + '; to max intensity in spectra')
        for i in range(0,A.shape[0]):
            if(np.amin(A[i]) <= 0):
                A[i,:] = A[i,:] - np.amin(A[i,:]) + 1e-8
            A[i,:] = np.multiply(A[i,:], preprocDef.YnormTo/A[i,A[i][YnormXind].tolist().index(max(A[i][YnormXind].tolist()))+YnormXind[0]])

    if preprocDef.StandardScalerFlag == True:
        print('  Using StandardScaler from sklearn ')
        A = preprocDef.scaler.fit_transform(A)

    #**********************************************
    ''' Energy normalization range '''
    #**********************************************
    if preprocDef.enRestrictRegion == True:
        A = A[:,range(preprocDef.enLim1, preprocDef.enLim2)]
        En = En[range(preprocDef.enLim1, preprocDef.enLim2)]
        Aorig = Aorig[:,range(preprocDef.enLim1, preprocDef.enLim2)]

        if type == 0:
            print( '  Restricting energy range between: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    else:
        if type == 0:
            if(preprocDef.cherryPickEnPoint == True):
                print( '  Using selected spectral points:')
                print(En)
            else:
                print( '  Using full energy range: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')

    return A, Cl, En, Aorig

#**********************************************************************************
''' Preprocess Prediction data '''
#**********************************************************************************
def preProcessNormPredData(R, Rx, En, YnormXind, type):
    print(' Processing Prediction data file... ')
    #**********************************************************************************
    ''' Reformat x-axis in case it does not match that of the training data '''
    #**********************************************************************************
    if(R.shape[0] != En.shape):
        if type == 0:
            print('\033[1m' + '  WARNING: Different number of datapoints for the x-axis\n  for training (' + str(En.shape) + ') and sample (' + str(R.shape[0]) + ') data.\n  Reformatting x-axis of sample data...\n' + '\033[0m')
        R = np.interp(En, Rx, R)
    R = R.reshape(1,-1)
    Rorig = np.copy(R)

    #**********************************************
    ''' Normalize/preprocess if flags are set '''
    #**********************************************
    if preprocDef.Ynorm == True:
        if type == 0:
            if preprocDef.fullYnorm == False:
                print('  Normalizing spectral intensity to: ' + str(preprocDef.YnormTo) + '; En = [' + str(preprocDef.YnormX-preprocDef.YnormXdelta) + ', ' + str(preprocDef.YnormX+preprocDef.YnormXdelta) + ']')
            else:
                print('  Normalizing spectral intensity to: ' + str(preprocDef.YnormTo) + '; to max intensity in spectra')
    
        if(np.amin(R) <= 0):
            print('  Spectra max below zero detected')
            R[0,:] = R[0,:] - np.amin(R[0,:]) + 1e-8
        R[0,:] = np.multiply(R[0,:], preprocDef.YnormTo/R[0,R[0][YnormXind].tolist().index(max(R[0][YnormXind].tolist()))+YnormXind[0]])

    if preprocDef.StandardScalerFlag == True:
        print('  Using StandardScaler from sklearn ')
        R = preprocDef.scaler.transform(R)
    
    #**********************************************
    ''' Energy normalization range '''
    #**********************************************
    if preprocDef.enRestrictRegion == True:
        #A = A[:,range(preprocDef.enLim1, preprocDef.enLim2)]
        En = En[range(preprocDef.enLim1, preprocDef.enLim2)]
        R = R[:,range(preprocDef.enLim1, preprocDef.enLim2)]
        
        if type == 0:
            print( '  Restricting energy range between: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    else:
        if type == 0:
            if(preprocDef.cherryPickEnPoint == True):
                print( '  Using selected spectral points:')
                print(En)
            else:
                print( '  Using full energy range: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    return R, Rorig

#**********************************************************************************
''' Preprocess prediction data '''
#**********************************************************************************
def preProcessNormMap(A, En, type):
    #**********************************************************************************
    ''' Reformat x-axis in case it does not match that of the training data '''
    #**********************************************************************************

    # Find index corresponding to energy value to be used for Y normalization
    if preprocDef.fullYnorm == False:
        YnormXind = np.where((En<float(preprocDef.YnormX+preprocDef.YnormXdelta)) & (En>float(preprocDef.YnormX-preprocDef.YnormXdelta)))[0].tolist()
    else:
        YnormXind = np.where(En>0)[0].tolist()
    Aorig = np.copy(A)

    #**********************************************
    ''' Normalize/preprocess if flags are set '''
    #**********************************************
    if preprocDef.Ynorm == True:
        if type == 0:
            print(' Normalizing spectral intensity to: ' + str(preprocDef.YnormTo) + '; En = [' + str(preprocDef.YnormX-preprocDef.YnormXdelta) + ', ' + str(preprocDef.YnormX+preprocDef.YnormXdelta) + ']')
        for i in range(0,A.shape[0]):
            A[i,:] = np.multiply(A[i,:], preprocDef.YnormTo/np.amax(A[i]))


    if preprocDef.StandardScalerFlag == True:
        print('  Using StandardScaler from sklearn ')
        A = preprocDef.scaler.fit_transform(A)

    #**********************************************
    ''' Energy normalization range '''
    #**********************************************
    if preprocDef.enRestrictRegion == True:
        A = A[:,range(preprocDef.enLim1, preprocDef.enLim2)]
        En = En[range(preprocDef.enLim1, preprocDef.enLim2)]
        Aorig = Aorig[:,range(preprocDef.enLim1, preprocDef.enLim2)]
        if type == 0:
            print( ' Restricting energy range between: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')
    else:
        if type == 0:
            print( ' Using full energy range: [' + str(En[0]) + ', ' + str(En[En.shape[0]-1]) + ']\n')

    return A, En, Aorig

####################################################################
''' Format subset of training data '''
####################################################################
def formatSubset(A, Cl, percent):
    from sklearn.model_selection import train_test_split
    A_train, A_cv, Cl_train, Cl_cv = \
    train_test_split(A, Cl, test_size=percent, random_state=42)
    return A_train, Cl_train, A_cv, Cl_cv

####################################################################
''' Open map files '''
####################################################################
def readPredMap(mapFile):
    try:
        with open(mapFile, 'r') as f:
            En = np.array(f.readline().split(), dtype=np.dtype(float))
            A = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Map data file not found \n' + '\033[0m')
        return

    X = A[:,0]
    Y = A[:,1]
    A = np.delete(A, np.s_[0:2], 1)
    print(' Shape map: ' + str(A.shape))
    return X, Y, A, En

####################################################################
''' Save map files '''
####################################################################
def saveMap(file, type, extension, s, x1, y1, comma):
    inputFile = saveMapName(file, type, extension, comma)

    with open(inputFile, "a") as coord_file:
        if comma==True:
            coord_file.write('{:},'.format(x1))
            coord_file.write('{:},'.format(y1))
        else:
            coord_file.write('{:}\t'.format(x1))
            coord_file.write('{:}\t'.format(y1))
        coord_file.write('{:}\n'.format(s))
        coord_file.close()

def saveMapName(file, type, extension, comma):
    if comma==True:
        extension2 = '_map.csv'
    else:
        extension2 = '_map.txt'
    return os.path.splitext(file)[0] + '_' + type + '-' + extension + extension2


#************************************
''' Plot Probabilities'''
#************************************
def plotProb(clf, R):
    prob = clf.predict_proba(R)[0].tolist()
    print(' Probabilities of this sample within each class: \n')
    for i in range(0,clf.classes_.shape[0]):
        print(' ' + str(clf.classes_[i]) + ': ' + str(round(100*prob[i],2)) + '%')
    import matplotlib.pyplot as plt
    print('\n Stand by: Plotting probabilities for each class... \n')
    plt.title('Probability density per class')
    for i in range(0, clf.classes_.shape[0]):
        plt.scatter(clf.classes_[i], round(100*prob[i],2), label='probability', c = 'red')
    plt.grid(True)
    plt.xlabel('Class')
    plt.ylabel('Probability [%]')
    plt.show()


#************************************
''' Plot Training data'''
#************************************
def plotTrainData(A, En, R, plotAllSpectra, learnFileRoot):
    import matplotlib.pyplot as plt
    if plotDef.plotAllSpectra == True:
        step = 1
        learnFileRoot = learnFileRoot + '_full-set'
    else:
        step = plotDef.stepSpectraPlot
        learnFileRoot = learnFileRoot + '_partial-' + str(step)

    print(' Plotting Training dataset in: ' + learnFileRoot + '.png\n')
    if preprocDef.Ynorm ==True:
        plt.title('Normalized Training Data')
    else:
        plt.title('Training Data')
    for i in range(0,A.shape[0], step):
        plt.plot(En, A[i,:], label='Training data')
    plt.plot(En, R[0,:], linewidth = 4, label='Sample data')
    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')

    plt.savefig(learnFileRoot + '.png', dpi = 160, format = 'png')  # Save plot

    if plotDef.showTrainingDataPlot == True:
        plt.show()

    plt.close()


#************************************
''' Plot Processed Maps'''
#************************************
def plotMaps(X, Y, A, label):
    print(' Plotting ' + label + ' Map...\n')
    import scipy.interpolate
    xi = np.linspace(min(X), max(X))
    yi = np.linspace(min(Y), max(Y))
    xi, yi = np.meshgrid(xi, yi)

    rbf = scipy.interpolate.Rbf(Y, -X, A, function='linear')
    zi = rbf(xi, yi)
    import matplotlib.pyplot as plt
    plt.imshow(zi, vmin=A.min(), vmax=A.max(), origin='lower',label='data',
               extent=[X.min(), X.max(), Y.min(), Y.max()])
    plt.title(label)
    plt.xlabel('X [um]')
    plt.ylabel('Y [um]')
    plt.show()


####################################################################
''' Make header, if absent, for the summary file '''
####################################################################
def makeHeaderSummary(file, learnFile):
    if os.path.isfile(file) == False:
        summaryHeader1 = ['Training File:', learnFile]
        summaryHeader2 = ['File','SVM-HC','SVM-Prob%', 'NN-HC', 'NN-Prob%', 'TF-HC', 'TF-Prob%', 'TF-Accuracy%']
        with open(file, "a") as sum_file:
            csv_out=csv.writer(sum_file)
            csv_out.writerow(summaryHeader1)
            csv_out.writerow(summaryHeader2)
            sum_file.close()

#************************************
''' Info on Classification Report '''
#************************************
def runClassReport(clf, A, Cl):
    from sklearn.metrics import classification_report
    y_pred = clf.predict(A)
    print(classification_report(Cl, y_pred, target_names=clf.classes_))
    print(' Precision is the probability that, given a classification result for a sample,\n' +
          ' the sample actually belongs to that class. Recall (Accuracy) is the probability that a \n' +
          ' sample will be correctly classified for a given class. f1-score combines both \n' +
          ' accuracy and precision to give a single measure of relevancy of the classifier results.\n')

#************************************
''' Introduce Noise in Data '''
#************************************
def scrambleNoise(A, offset):
    from random import uniform
    for i in range(A.shape[1]):
        A[:,i] += offset*uniform(-1,1)

