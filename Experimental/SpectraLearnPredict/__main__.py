#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict
* Perform Machine Learning on Spectroscopy Data.
* Version 20171007d
*
* Uses: Deep Neural Networks, TensorFlow, SVM, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
print(__doc__)

import matplotlib
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Agg')

import numpy as np
import sys, os.path, getopt, glob, csv
import random, time, configparser, os
from os.path import exists, splitext
from os import rename
from datetime import datetime, date

from .slp import *

#**********************************************
''' Main '''
#**********************************************
def main():

    start_time = time.clock()
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "fambkph:", ["file", "accuracy", "map", "batch",
                                   "kmaps", "pca", "help"])
    except:
        usage()
        sys.exit(2)

    if opts == []:
        usage()
        sys.exit(2)

    print(" Using training file: ", sys.argv[2],"\n")
    for o, a in opts:
        if o in ("-f" , "--file"):
            try:
                LearnPredictFile(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)

        if o in ("-a" , "--accuracy"):
            print('\033[1m Running in cross validation mode for accuracy determination...\033[0m\n')
            try:
                if sys.argv[3]:
                    testFile = sys.argv[3]
            except:
                preprocDef.subsetCrossValid = True
                testFile = "tmp"
            #try:
            trainAccuracy(sys.argv[2], testFile)
            #except:
            #    usage()
            #    sys.exit(2)

        if o in ("-m" , "--map"):
            try:
                LearnPredictMap(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)

        if o in ("-b" , "--batch"):
            try:
                LearnPredictBatch(sys.argv[2])
            except:
                usage()
                sys.exit(2)

        if o in ("-p" , "--pca"):
            if len(sys.argv) > 3:
                numPCAcomp = int(sys.argv[3])
            else:
                numPCAcomp = pcaDef.numPCAcomponents
            try:
                runPCA(sys.argv[2], numPCAcomp)
            except:
                usage()
                sys.exit(2)

        if o in ("-k" , "--kmaps"):
            if len(sys.argv) > 3:
                numKMcomp = int(sys.argv[3])
            else:
                numKMcomp = kmDef.numKMcomponents
            try:
                KmMap(sys.argv[2], numKMcomp)
            except:
                usage()
                sys.exit(2)
        total_time = time.clock() - start_time
        print(" Total time (s):",total_time)

#**********************************************
''' Learn and Predict - File'''
#**********************************************
def LearnPredictFile(learnFile, sampleFile):
    ''' Open and process training data '''
    En, Cl, A, YnormXind = readLearnFile(learnFile)

    learnFileRoot = os.path.splitext(learnFile)[0]

    ''' Run PCA '''
    if pcaDef.runPCA == True:
        runPCAmain(A, Cl, En)

    ''' Open prediction file '''
    R, Rx = readPredFile(sampleFile)

    ''' Preprocess prediction data '''
    A, Cl, En, Aorig = preProcessNormLearningData(A, En, Cl, YnormXind, 0)
    R, Rorig = preProcessNormPredData(R, Rx, A, En, Cl, YnormXind, 0)
    
    ''' Run Neural Network - TensorFlow'''
    if dnntfDef.runDNNTF == True:
        if dnntfDef.runSkflowDNNTF == True:
            clf_dnntf, le_dnntf  = trainDNNTF(A, Cl, A, Cl, learnFileRoot)
            predDNNTF(clf_dnntf, le_dnntf, R, Cl)
        else:
            clf_dnntf, le_dnntf  = trainDNNTF2(A, Cl, A, Cl, learnFileRoot)
            predDNNTF2(clf_dnntf, le_dnntf, R, Cl)
    
    ''' Run Neural Network - sklearn'''
    if nnDef.runNN == True:
        clf_nn = trainNN(A, Cl, A, Cl, learnFileRoot)
        predNN(clf_nn, A, Cl, R)

    ''' Run Support Vector Machines '''
    if svmDef.runSVM == True:
        clf_svm = trainSVM(A, Cl, A, Cl, learnFileRoot)
        predSVM(clf_svm, A, Cl, R)

    ''' Tensorflow '''
    if tfDef.runTF == True:
        trainTF(A, Cl, A, Cl, learnFileRoot)
        predTF(A, Cl, R, learnFileRoot)

    ''' Plot Training Data '''
    if plotDef.createTrainingDataPlot == True:
        plotTrainData(A, En, R, plotDef.plotAllSpectra, learnFileRoot)

    ''' Run K-Means '''
    if kmDef.runKM == True:
        runKMmain(A, Cl, En, R, Aorig, Rorig)


#**********************************************
''' Train and accuracy'''
#**********************************************
def trainAccuracy(learnFile, testFile):
    ''' Open and process training data '''

    En, Cl, A, YnormXind = readLearnFile(learnFile)
    
    if preprocDef.subsetCrossValid == True:
        print(" Cross-validation training using: ",str(preprocDef.percentCrossValid*100),
              "% of training file as test subset\n")

        A, Cl, A_test, Cl_test = formatSubset(A, Cl, preprocDef.percentCrossValid)
        En_test = En
    else:
        print(" Cross-validation training using: privided test subset (",testFile,")\n")
        En_test, Cl_test, A_test, YnormXind2 = readLearnFile(testFile)
    
    learnFileRoot = os.path.splitext(learnFile)[0]
    
    ''' Plot Training Data - Raw '''
    if plotDef.createTrainingDataPlot == True:
        plotTrainData(A, En, A_test, plotDef.plotAllSpectra, learnFileRoot+"_raw")
    
    ''' Preprocess prediction data '''
    A, Cl, En, Aorig = preProcessNormLearningData(A, En, Cl, YnormXind, 0)
    A_test, Cl_test, En_test, Aorig_test = preProcessNormLearningData(A_test, En_test, Cl_test, YnormXind, 0)
    
    ''' Run Neural Network - TensorFlow'''
    if dnntfDef.runDNNTF == True:
        if dnntfDef.runSkflowDNNTF == True:
            clf_dnntf, le_dnntf  = trainDNNTF(A, Cl, A_test, Cl_test, learnFileRoot)
        else:
            clf_dnntf, le_dnntf  = trainDNNTF2(A, Cl, A_test, Cl_test, learnFileRoot)

    ''' Run Neural Network - sklearn'''
    if nnDef.runNN == True:
        clf_nn = trainNN(A, Cl, A_test, Cl_test, learnFileRoot)
    
    ''' Run Support Vector Machines '''
    if svmDef.runSVM == True:
        clf_svm = trainSVM(A, Cl, A_test, Cl_test, learnFileRoot)
    
    ''' Tensorflow '''
    if tfDef.runTF == True:
        trainTF(A, Cl, A_test, Cl_test, learnFileRoot)
    
    ''' Plot Training Data - Normalized'''
    if plotDef.createTrainingDataPlot == True:
        plotTrainData(A, En, A_test, plotDef.plotAllSpectra, learnFileRoot+"_norm")

#**********************************************
''' Process - Batch'''
#**********************************************
def LearnPredictBatch(learnFile):
    summary_filename = 'summary' + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
    makeHeaderSummary(summary_filename, learnFile)
    ''' Open and process training data '''
    En, Cl, A, YnormXind = readLearnFile(learnFile)
    A, Cl, En, Aorig = preProcessNormLearningData(A, En, Cl, YnormXind, 0)
    
    if multiproc == True:
        import multiprocessing as mp
        p = mp.Pool()
        for f in glob.glob('*.txt'):
            if (f != learnFile):
                p.apply_async(processSingleBatch, args=(f, En, Cl, A, Aorig, YnormXind, summary_filename, learnFile))
        p.close()
        p.join()
    else:
        for f in glob.glob('*.txt'):
            if (f != learnFile):
                processSingleBatch(f, En, Cl, A, Aorig, YnormXind, summary_filename, learnFile)

def processSingleBatch(f, En, Cl, A, Aorig, YnormXind, summary_filename, learnFile):
    print(' Processing file: \033[1m' + f + '\033[0m\n')
    R, Rx = readPredFile(f)
    summaryFile = [f]
    ''' Preprocess prediction data '''
    R, Rorig = preProcessNormPredData(R, Rx, A, En, Cl, YnormXind, 0)

    learnFileRoot = os.path.splitext(learnFile)[0]
    
    ''' Run Neural Network - TensorFlow'''
    if dnntfDef.runDNNTF == True:
        if dnntfDef.runSkflowDNNTF == True:
            clf_dnntf, le_dnntf  = trainDNNTF(A, Cl, A, Cl, learnFileRoot)
            dnntfPred, dnntfProb = predDNNTF(clf_dnntf, le_dnntf, R, Cl)
        else:
            clf_dnntf, le_dnntf  = trainDNNTF2(A, Cl, A, Cl, learnFileRoot)
            dnntfPred, dnntfProb = predDNNTF2(clf_dnntf, le_dnntf, R, Cl)
        summaryFile.extend([nnPred, nnProb])
        dnntfDef.alwaysRetrain = False
    
    ''' Run Neural Network - sklearn'''
    if nnDef.runNN == True:
        clf_nn = trainNN(A, Cl, A, Cl, learnFileRoot)
        nnPred, nnProb = predNN(clf_nn, A, Cl, R)
        summaryFile.extend([nnPred, nnProb])
        nnDef.alwaysRetrain = False

    ''' Run Support Vector Machines '''
    if svmDef.runSVM == True:
        clf_svm = trainSVM(A, Cl, A, Cl, learnFileRoot)
        svmPred, svmProb = predSVM(clf_svm, A, Cl, En, R)
        summaryFile.extend([svmPred, svmProb])
        svmDef.alwaysRetrain = False

    ''' Tensorflow '''
    if tfDef.runTF == True:
        trainTF(A, Cl, A, Cl, learnFileRoot)
        tfPred, tfProb = predTF(A, Cl, R, learnFileRoot)
        summaryFile.extend([tfPred, tfProb, tfAccur])
        tfDef.tfalwaysRetrain = False

    ''' Run K-Means '''
    if kmDef.runKM == True:
        kmDef.plotKM = False
        kmPred = runKMmain(A, Cl, En, R, Aorig, Rorig)
        summaryFile.extend([kmPred])

    with open(summary_filename, "a") as sum_file:
        csv_out=csv.writer(sum_file)
        csv_out.writerow(summaryFile)
        sum_file.close()

#**********************************************
''' Learn and Predict - Maps'''
#**********************************************
def LearnPredictMap(learnFile, mapFile):
    ''' Open and process training data '''
    En, Cl, A, YnormXind = readLearnFile(learnFile)

    learnFileRoot = os.path.splitext(learnFile)[0]

    ''' Open prediction map '''
    X, Y, R, Rx = readPredMap(mapFile)
    type = 0
    i = 0;
    svmPred = nnPred = tfPred = kmPred = np.empty([X.shape[0]])
    A, Cl, En, Aorig = preProcessNormLearningData(A, En, Cl, YnormXind, type)
    print(' Processing map...' )
    
    if nnDef.runNN == True:
        clf_nn = trainNN(A, Cl, A, Cl, learnFileRoot)

    if dnntfDef.runDNNTF == True:
        if dnntfDef.runSkflowDNNTF == True:
            clf_dnntf, le_dnntf  = trainDNNTF(A, Cl, A, Cl, learnFileRoot)
        else:
            clf_dnntf, le_dnntf  = trainDNNTF2(A, Cl, A, Cl, learnFileRoot)

    if svmDef.runSVM == True:
        clf_svm = trainSVM(A, Cl, A, Cl, learnFileRoot)

    for r in R[:]:
        r, rorig = preProcessNormPredData(r, Rx, A, En, Cl, YnormXind, type)
        type = 1
        
        ''' Run Neural Network - TensorFlow'''
        if dnntfDef.runDNNTF == True:
            if dnntfDef.runSkflowDNNTF == True:
                dnntfPred[i], temp = predDNNTF(cl_dnntf, le_dnntf, r, Cl)
            else:
                dnntfPred[i], temp = predDNNTF2(cl_dnntf, le_dnntf, r, Cl)
            
            saveMap(mapFile, 'DNN-TF', 'HC', dnntfPred[i], X[i], Y[i], True)
            dnnDef.alwaysRetrain = False
        
        ''' Run Neural Network - sklearn'''
        if nnDef.runNN == True:
            nnPred[i], temp = predNN(clf_nn, A, Cl, r)
            saveMap(mapFile, 'NN', 'HC', nnPred[i], X[i], Y[i], True)
            nnDef.alwaysRetrain = False
        
        ''' Run Support Vector Machines '''
        if svmDef.runSVM == True:
            svmPred[i], temp = predSVM(clf_svm, A, Cl, En, r)
            saveMap(mapFile, 'svm', 'HC', svmPred[i], X[i], Y[i], True)
            svmDef.alwaysRetrain = False

        ''' Tensorflow '''
        if tfDef.runTF == True:
            trainTF(A, Cl, A, Cl, learnFileRoot)
            tfPred, temp = predTF(A, Cl, r, learnFileRoot)
            saveMap(mapFile, 'TF', 'HC', tfPred[i], X[i], Y[i], True)
            tfDef.alwaysRetrain = False

        ''' Run K-Means '''
        if kmDef.runKM == True:
            kmDef.plotKM = False
            kmPred[i] = runKMmain(A, Cl, En, r, Aorig, rorig)
            saveMap(mapFile, 'KM', 'HC', kmPred[i], X[i], Y[i], True)
        i+=1

    if dnntfDef.plotMap == True and dnntfDef.runDNNTF == True:
        plotMaps(X, Y, dnntfPred, 'Deep Neural networks - tensorFlow')
    if nnDef.plotMap == True and nnDef.runNN == True:
        plotMaps(X, Y, nnPred, 'Deep Neural networks - sklearn')
    if svmDef.plotMap == True and svmDef.runSVM == True:
        plotMaps(X, Y, svmPred, 'SVM')
    if tfDef.plotMap == True and tfDef.runTF == True:
        plotMaps(X, Y, tfPred, 'TensorFlow')
    if kmDef.plotMap == True and kmDef.runKM == True:
        plotMaps(X, Y, kmPred, 'K-Means Prediction')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
