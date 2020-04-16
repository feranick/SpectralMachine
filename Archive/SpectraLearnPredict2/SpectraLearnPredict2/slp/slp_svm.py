#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict2 - SVM
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
import sys, os.path, getopt, glob, csv, pickle
import random, time, configparser, os
from os.path import exists, splitext
from os import rename
from datetime import datetime, date

from .slp_config import *

#********************************************************************************
''' Support Vector Machines - SVM '''
''' http://scikit-learn.org/stable/modules/svm.html '''
#********************************************************************************
''' Train SVM '''
#********************************************************************************
def trainSVM(A, Cl, A_test, Cl_test, Root):
    from sklearn import svm
    from sklearn.externals import joblib
    from sklearn import preprocessing
    
    print('==========================================================================\n')
    print('\033[1m Running Support Vector Machine (kernel: ' + svmDef.kernel + ')\033[0m')
    
    svmTrainedData = Root + '.svmModel.pkl'
    model_le = Root + '.svmLabelEnc.pkl'

    le = preprocessing.LabelEncoder()
    try:
        if svmDef.alwaysRetrain == False:
            with open(svmTrainedData):
                print('  Opening SVM training model...\n')
                clf = joblib.load(svmTrainedData)
                le = joblib.load(model_le)
        else:
            raise ValueError('  Force retraining SVM model')
    except:
        totA = np.vstack((A, A_test))
        totCl = np.append(Cl, Cl_test)
        totCl2 = le.fit_transform(totCl)
        Cl2 = le.transform(Cl)
        Cl2_test = le.transform(Cl_test)
        
        #**********************************************
        ''' Retrain training model if not available'''
        #**********************************************
        print('  Retraining SVM data...')
        clf = svm.SVC(C = svmDef.Cfactor, decision_function_shape = 'ovr', probability=True, gamma='auto')
        
        print("  Training on the full training dataset\n")
        clf.fit(A,Cl2)
        accur = clf.score(A_test,Cl2_test)
        print('  Mean accuracy: ',100*accur,'%')

        Z = clf.decision_function(A)
        print('\n  Number of classes = ' + str(Z.shape[1]))
        joblib.dump(clf, svmTrainedData)
        joblib.dump(le, model_le)
        if svmDef.showClasses:
            print('  List of classes: ' + str(clf.classes_))

    print('==========================================================================\n')
    return clf, le

#********************************************************************************
''' Predict using SVM '''
#********************************************************************************
def predSVM(clf, A, Cl, R, le):
    R_pred_le = clf.predict(R)
    prob = clf.predict_proba(R)[0].tolist()

    if R_pred_le.size >0:
        R_pred = le.inverse_transform(R_pred_le)
    else:
        R_pred = 0
    rosterPred = np.where(clf.predict_proba(R)[0]>svmDef.thresholdProbabilitySVMPred/100)[0]

    print('  ================================')
    print('  \033[1mSVM\033[0m - Probability >',str(svmDef.thresholdProbabilitySVMPred),'%')
    print('  ================================')
    print('  Prediction\t| Probability [%]')
    for i in range(rosterPred.shape[0]):
        print("  {0:.2f}\t\t| {1:.1f}".format(np.unique(Cl)[rosterPred][i],100*clf.predict_proba(R)[0][rosterPred][i]))
    print('  ================================')

    print('\033[1m' + '\n Predicted value (SVM) = ' + str(R_pred[0]) + ' (probability = ' +
      str(round(100*max(prob),1)) + '%)\033[0m\n')
    
    #**************************************
    ''' SVM Classification Report '''
    #**************************************
    if svmDef.svmClassReport:
        print(' SVM Classification Report \n')
        runClassReport(clf, A, Cl)

    #*************************
    ''' Plot probabilities '''
    #*************************
    if plotDef.showProbPlot:
        plotProb(clf, R)

    return R_pred[0], round(100*max(prob),1)

#********************************************************************************
''' Run PCA '''
''' Transform data:
    pca.fit(data).transform(data)
    Loading Vectors (eigenvectors):
    pca.components_
    Eigenvalues:
    pca.explained_variance_ratio
    '''
#********************************************************************************
def runPCA(learnFile, numPCAcomponents):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from matplotlib import cm

    ''' Open and process training data '''
    En, Cl, A, YnormXind = readLearnFile(learnFile)

    print('==========================================================================\n')
    print(' Running PCA...\n')
    print(' Number of unique identifiers in training data: ' + str(np.unique(Cl).shape[0]))
    if pcaDef.customNumPCAComp == False:
        numPCAcomp = np.unique(Cl).shape[0]
    else:
        numPCAcomp = numPCAcomponents
    print(' Number of Principal components: ' + str(numPCAcomp) + '\n')
    pca = PCA(n_components=numPCAcomp)
    A_r = pca.fit(A).transform(A)

    for i in range(0,pca.components_.shape[0]):
        print(' Score PC ' + str(i) + ': ' + '{0:.0f}%'.format(pca.explained_variance_ratio_[i] * 100))
    print('')

    if plotDef.showPCAPlots:
        print(' Plotting Loadings and score plots... \n')

        #***************************
        ''' Plotting Loadings '''
        #***************************
        for i in range(0,pca.components_.shape[0]):
            plt.plot(En, pca.components_[i,:], label='PC' + str(i) + ' ({0:.0f}%)'.format(pca.explained_variance_ratio_[i] * 100))
        plt.plot((En[0], En[En.shape[0]-1]), (0.0, 0.0), 'k--')
        plt.title('Loadings plot')
        plt.xlabel('Raman shift [1/cm]')
        plt.ylabel('Principal component')
        plt.legend()
        plt.figure()

        #***************************
        ''' Plotting Scores '''
        #***************************
        Cl_ind = np.zeros(len(Cl))
        Cl_labels = np.zeros(0)
        ind = np.zeros(np.unique(Cl).shape[0])

        for i in range(len(Cl)):
            if (np.in1d(Cl[i], Cl_labels, invert=True)):
                Cl_labels = np.append(Cl_labels, Cl[i])

        for i in range(len(Cl)):
            Cl_ind[i] = np.where(Cl_labels == Cl[i])[0][0]

            colors = [ cm.jet(x) for x in np.linspace(0, 1, ind.shape[0]) ]

        for color, i, target_name in zip(colors, range(ind.shape[0]), Cl_labels):
            plt.scatter(A_r[Cl_ind==i,0], A_r[Cl_ind==i,1], color=color, alpha=.8, lw=2, label=target_name)

        plt.title('Score plot')
        plt.xlabel('PC 0 ({0:.0f}%)'.format(pca.explained_variance_ratio_[0] * 100))
        plt.ylabel('PC 1 ({0:.0f}%)'.format(pca.explained_variance_ratio_[1] * 100))
        plt.figure()

        plt.title('Score box plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Score')
        for j in range(pca.components_.shape[0]):
            for color, i, target_name in zip(colors, range(ind.shape[0]), Cl_labels):
                plt.scatter([j+1]*len(A_r[Cl_ind==i,j]), A_r[Cl_ind==i,j], color=color, alpha=.8, lw=2, label=target_name)
        plt.boxplot(A_r)
        plt.figure()

        #******************************
        ''' Plotting Scores vs H:C '''
        #******************************
        for j in range(pca.components_.shape[0]):
            for color, i, target_name in zip(colors, range(ind.shape[0]), Cl_labels):
                plt.scatter(np.asarray(Cl)[Cl_ind==i], A_r[Cl_ind==i,j], color=color, alpha=.8, lw=2, label=target_name)
            plt.xlabel('H:C elemental ratio')
            plt.ylabel('PC ' + str(j) + ' ({0:.0f}%)'.format(pca.explained_variance_ratio_[j] * 100))
            plt.figure()
        plt.show()

