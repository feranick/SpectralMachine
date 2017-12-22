#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict
* Perform Machine Learning on Spectroscopy Data.
* version: 20171222c
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

#***************************************************************
''' Parameters and configuration  '''
#***************************************************************
class Configuration():
    def __init__(self):
        self.configFile = os.getcwd()+"/SpectraLearnPredict.ini"
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print("Configuration file does not exist: Creating one.")
            self.createConfig()

    # Hadrcoded default definitions for the confoguration file
    def preprocDef(self):
        self.conf['Preprocessing'] = {
            'Ynorm' : True,
            'fullYnorm' : True,
            'StandardScalerFlag' : False,
            'subsetCrossValid' : False,
            'percentCrossValid' : 0.05,
            'YnormTo' : 1,
            'YnormX' : 1600,
            'YnormXdelta' : 30,
            'enRestrictRegion' : False,
            'enLim1' : 450,
            'enLim2' : 550,
            'scrambleNoiseFlag' : False,
            'scrambleNoiseOffset' : 0.1,
            'cherryPickEnPoint' : False,
            'enSel' : [1050, 1150, 1220, 1270, 1330, 1410, 1480, 1590, 1620, 1650],
            'enSelDelta' : [2, 2, 2, 2, 10, 2, 2, 15, 5, 2],
            }

    def dnntfDef(self):
        self.conf['DNNClassifier'] = {
            'runDNNTF' : True,
            'runSkflowDNNTF' : True,
            'alwaysRetrainDNNTF' : False,
            'alwaysImproveDNNTF' : True,
            'hidden_layersDNNTF' : [400,],
            'optimizerDNNTF' : "ProximalAdagrad",
            'learning_rateDNNTF' : 0.1,
            'l2_reg_strengthDNNTF' : 1e-4,
            'activation_functionDNNTF' : "tanh",
            'dropout_percDNNTF' : str(None),
            'trainingStepsDNNTF' : 1000,
            'valMonitorSecsDNNTF' : 200,
            'logCheckpointDNNTF' : True,
            'timeCheckpointDNNTF' : 20,
            'thresholdProbabilityPredDNNTF' : 0.01,
            'plotMapDNNTF' : True,
            'shuffleTrainDNNTF' : True,
            'shuffleTestDNNTF' : False,
            }
    
    def nnDef(self):
        self.conf['NNSklearn'] = {
            'runNN' : True,
            'alwaysRetrainNN' : False,
            'hidden_layersNN' : (400),
            'optimizerNN' : "adam",
            'activation_functionNN' : "tanh",
            'l2_reg_strengthNN' : 1e-4,
            'MLPRegressorNN' : False,
            'thresholdProbabilityPredNN' : 0.001,
            'plotMapNN' : True,
            'classReportNN' : False,
            }

    def svmDef(self):
        self.conf['SVM'] = {
            'runSVM' : True,
            'alwaysRetrainSVM' : False,
            'thresholdProbabilityPredSVM' : 3,
            'CfactorSVM' : 20,
            'kernelSVM' : "rbf",
            'showClassesSVM' : False,
            'plotMapSVM' : True,
            'classReportSVM' : False,
            }

    def pcaDef(self):
        self.conf['PCA'] = {
            'runPCA' : False,
            'customNumCompPCA' : True,
            'numComponentsPCA' : 2,
            }

    def kmDef(self):
        self.conf['KMeans'] = {
            'runKM' : False,
            'customNumCompKM' : False,
            'numComponentsKM' : 20,
            'plotKM' : False,
            'plotMapKM' : True,
            }

    def tfDef(self):
        self.conf['TensorFlow'] = {
            'runTF' : False,
            'alwaysRetrainTF' : True,
            'alwaysImproveTF' : True,
            'thresholdProbabilityPredTF' : 30,
            'decayLearnRateTF' : True,
            'learnRateTF' : 0.75,
            'subsetCrossValidTF' : False,
            'percentCrossValidTF' : 0.05,
            'logCheckpointTF' : False,
            'plotMapTF' : True,
            'plotClassDistribTF' : False,
            'enableTensorboardTF' : False,
            }

    def plotDef(self):
        self.conf['Plotting'] = {
            'showProbPlot' : False,
            'showPCAPlots' : True,
            'createTrainingDataPlot' : False,
            'showTrainingDataPlot' : False,
            'plotAllSpectra' : True,
            'stepSpectraPlot' : 100,
            }

    def sysDef(self):
        self.conf['System'] = {
            'multiproc' : False,
            }

    # Read configuration file into usable variables
    def readConfig(self, configFile):
        self.conf.read(configFile)
        
        self.preprocDef = self.conf['Preprocessing']
        self.dnntfDef = self.conf['DNNClassifier']
        self.nnDef = self.conf['NNSklearn']
        self.svmDef = self.conf['SVM']
        self.pcaDef = self.conf['PCA']
        self.kmDef = self.conf['KMeans']
        self.tfDef = self.conf['TensorFlow']
        self.plotDef = self.conf['Plotting']
        self.sysDef = self.conf['System']
        
        self.Ynorm = self.conf.getboolean('Preprocessing','Ynorm')
        self.fullYnorm = self.conf.getboolean('Preprocessing','fullYnorm')
        self.StandardScalerFlag = self.conf.getboolean('Preprocessing','StandardScalerFlag')
        self.subsetCrossValid = self.conf.getboolean('Preprocessing','subsetCrossValid')
        self.percentCrossValid = self.conf.getfloat('Preprocessing','percentCrossValid')
        self.YnormTo = self.conf.getfloat('Preprocessing','YnormTo')
        self.YnormX = self.conf.getfloat('Preprocessing','YnormX')
        self.YnormXdelta = self.conf.getfloat('Preprocessing','YnormXdelta')
        self.enRestrictRegion = self.conf.getboolean('Preprocessing','enRestrictRegion')
        self.enLim1 = self.conf.getint('Preprocessing','enLim1')
        self.enLim2 = self.conf.getint('Preprocessing','enLim2')
        self.scrambleNoiseFlag = self.conf.getboolean('Preprocessing','scrambleNoiseFlag')
        self.scrambleNoiseOffset = self.conf.getfloat('Preprocessing','scrambleNoiseOffset')
        self.cherryPickEnPoint = self.conf.getboolean('Preprocessing','cherryPickEnPoint')
        self.enSel = eval(self.preprocDef['enSel'])
        self.enSelDelta = eval(self.preprocDef['enSelDelta'])
        
        self.runDNNTF = self.conf.getboolean('DNNClassifier','runDNNTF')
        self.runSkflowDNNTF = self.conf.getboolean('DNNClassifier','runSkflowDNNTF')
        self.alwaysRetrainDNNTF = self.conf.getboolean('DNNClassifier','alwaysRetrainDNNTF')
        self.alwaysImproveDNNTF = self.conf.getboolean('DNNClassifier','alwaysImproveDNNTF')
        self.hidden_layersDNNTF = eval(self.dnntfDef['hidden_layersDNNTF'])
        self.optimizerDNNTF = self.dnntfDef['optimizerDNNTF']
        self.learning_rateDNNTF = self.conf.getfloat('DNNClassifier','learning_rateDNNTF')
        self.l2_reg_strengthDNNTF = self.conf.getfloat('DNNClassifier','l2_reg_strengthDNNTF')
        self.activation_functionDNNTF = self.dnntfDef['activation_functionDNNTF']
        self.dropout_percDNNTF = eval(self.dnntfDef['dropout_percDNNTF'])
        self.trainingStepsDNNTF = self.conf.getint('DNNClassifier','trainingStepsDNNTF')
        self.valMonitorSecsDNNTF = self.conf.getint('DNNClassifier','valMonitorSecsDNNTF')
        self.logCheckpointDNNTF = self.conf.getboolean('DNNClassifier','logCheckpointDNNTF')
        self.timeCheckpointDNNTF = self.conf.getint('DNNClassifier','timeCheckpointDNNTF')
        self.thresholdProbabilityPredDNNTF = self.conf.getfloat('DNNClassifier','thresholdProbabilityPredDNNTF')
        self.plotMapDNNTF = self.conf.getboolean('DNNClassifier','plotMapDNNTF')
        try:
            self.shuffleTrainDNNTF = self.conf.getboolean('DNNClassifier','shuffleTrainDNNTF')
            self.shuffleTestDNNTF = self.conf.getboolean('DNNClassifier','shuffleTestDNNTF')
        except:
            self.shuffleTrainDNNTF = True
            self.shuffleTestDNNTF = False
        
        self.runNN = self.conf.getboolean('NNSklearn','runNN')
        self.alwaysRetrainNN = self.conf.getboolean('NNSklearn','alwaysRetrainNN')
        self.hidden_layersNN = eval(self.nnDef['hidden_layersNN'])
        self.optimizerNN = self.nnDef['optimizerNN']
        self.activation_functionNN = self.nnDef['activation_functionNN']
        self.l2_reg_strengthNN = self.conf.getfloat('NNSklearn','l2_reg_strengthNN')
        self.MLPRegressorNN = self.conf.getboolean('NNSklearn','MLPRegressorNN')
        self.thresholdProbabilityPredNN = self.conf.getfloat('NNSklearn','thresholdProbabilityPredNN')
        self.plotMapNN = self.conf.getboolean('NNSklearn','plotMapNN')
        self.classReportNN = self.conf.getboolean('NNSklearn','classReportNN')
    
        self.runSVM = self.conf.getboolean('SVM','runSVM')
        self.alwaysRetrainSVM = self.conf.getboolean('SVM','alwaysRetrainSVM')
        self.thresholdProbabilityPredSVM = self.conf.getfloat('SVM','thresholdProbabilityPredSVM')
        self.CfactorSVM = self.conf.getfloat('SVM','CfactorSVM')
        self.kernelSVM = self.svmDef['kernelSVM']
        self.showClassesSVM = self.conf.getboolean('SVM','showClassesSVM')
        self.plotMapSVM = self.conf.getboolean('SVM','plotMapSVM')
        self.classReportSVM = self.conf.getboolean('SVM','classReportSVM')
        
        self.runPCA = self.conf.getboolean('PCA','runPCA')
        self.customNumCompPCA = self.conf.getboolean('PCA','customNumCompPCA')
        self.numComponentsPCA = self.conf.getint('PCA','numComponentsPCA')

        self.runKM = self.conf.getboolean('KMeans','runKM')
        self.customNumCompKM = self.conf.getboolean('KMeans','customNumCompKM')
        self.numComponentsKM = self.conf.getint('KMeans','numComponentsKM')
        self.plotKM = self.conf.getboolean('KMeans','plotKM')
        self.plotMapKM = self.conf.getboolean('KMeans','plotMapKM')
        
        self.runTF = self.conf.getboolean('TensorFlow','runTF')
        self.alwaysRetrainTF = self.conf.getboolean('TensorFlow','alwaysRetrainTF')
        self.alwaysImproveTF = self.conf.getboolean('TensorFlow','alwaysImproveTF')
        self.thresholdProbabilityPredTF = self.conf.getfloat('TensorFlow','thresholdProbabilityPredTF')
        self.decayLearnRateTF = self.conf.getboolean('TensorFlow','decayLearnRateTF')
        self.learnRateTF = self.conf.getfloat('TensorFlow','learnRateTF')
        self.subsetCrossValidTF = self.conf.getboolean('TensorFlow','subsetCrossValidTF')
        self.percentCrossValidTF = self.conf.getfloat('TensorFlow','percentCrossValidTF')
        self.logCheckpointTF = self.conf.getboolean('TensorFlow','logCheckpointTF')
        self.plotMapTF = self.conf.getboolean('TensorFlow','plotMapTF')
        self.plotClassDistribTF = self.conf.getboolean('TensorFlow','plotClassDistribTF')
        self.enableTensorboardTF = self.conf.getboolean('TensorFlow','enableTensorboardTF')
        
        self.showProbPlot = self.conf.getboolean('Plotting','showProbPlot')
        self.showPCAPlots = self.conf.getboolean('Plotting','showPCAPlots')
        self.createTrainingDataPlot = self.conf.getboolean('Plotting','createTrainingDataPlot')
        self.showTrainingDataPlot = self.conf.getboolean('Plotting','showTrainingDataPlot')
        self.plotAllSpectra = self.conf.getboolean('Plotting','plotAllSpectra')
        self.stepSpectraPlot = self.conf.getint('Plotting','stepSpectraPlot')

        self.multiproc = self.conf.getboolean('System','multiproc')


    # Create configuration file
    def createConfig(self):
        try:
            self.preprocDef()
            self.dnntfDef()
            self.nnDef()
            self.svmDef()
            self.pcaDef()
            self.kmDef()
            self.tfDef()
            self.plotDef()
            self.sysDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except:
            print("Error in creating configuration file")

#***************************************************************
''' Spectra normalization, preprocessing, model selection  '''
#***************************************************************
class preprocDef:
    config = Configuration()
    config.readConfig(config.configFile)
    
    Ynorm = config.Ynorm   # Normalize spectra ("True": recommended)
    fullYnorm = config.fullYnorm  # Normalize considering full range ("True": recommended)
    StandardScalerFlag = config.StandardScalerFlag  # Standardize features by removing the mean and scaling to unit variance (sklearn)

    subsetCrossValid = config.subsetCrossValid
    percentCrossValid = config.percentCrossValid  # proportion of TEST data for cross validation

    YnormTo = config.YnormTo
    YnormX = config.YnormX
    YnormXdelta = config.YnormXdelta

    enRestrictRegion = config.enRestrictRegion
    enLim1 = config.enLim1 # for now use indexes rather than actual Energy
    enLim2 = config.enLim2    # for now use indexes rather than actual Energy
    
    scrambleNoiseFlag = config.scrambleNoiseFlag # Adds random noise to spectra (False: recommended)
    scrambleNoiseOffset = config.scrambleNoiseOffset

    if StandardScalerFlag == True:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()

    #**********************************************
    ''' Calculation by limited number of points '''
    #**********************************************
    cherryPickEnPoint = config.cherryPickEnPoint  # False recommended

    enSel = config.enSel
    enSelDelta = config.enSelDelta
    
    #enSel = [1220, 1270, 1590]
    #enSelDelta = [2, 2, 30]

    if(cherryPickEnPoint == True):
        enRestrictRegion = False
        print(' Calculation by limited number of points: ENABLED ')
        print(' THIS IS AN EXPERIMENTAL FEATURE \n')
        print(' Restricted range: DISABLED')

#***********************************************************
''' Deep Neural Networks - tensorflow via DNNClassifier'''
#***********************************************************
class dnntfDef:
    config = Configuration()
    config.readConfig(config.configFile)
    
    runDNNTF = config.runDNNTF
    runSkflowDNNTF = config.runSkflowDNNTF
    alwaysRetrain = config.alwaysRetrainDNNTF
    alwaysImprove = config.alwaysImproveDNNTF
    
    # Format: [number_neurons_HL1, number_neurons_HL2, number_neurons_HL3,...]
    hidden_layers = config.hidden_layersDNNTF

    # Stock Optimizers: Adagrad (recommended), Adam, Ftrl, RMSProp, SGD
    # https://www.tensorflow.org/api_guides/python/train
    #optimizer = "Adagrad"

    # Additional optimizers: ProximalAdagrad, AdamOpt, Adadelta,
    #                        GradientDescent, ProximalGradientDescent
    # https://www.tensorflow.org/api_guides/python/train
    optimizer = config.optimizerDNNTF
    
    learning_rate = config.learning_rateDNNTF
    l2_reg_strength = config.l2_reg_strengthDNNTF
    
    # activation functions: https://www.tensorflow.org/api_guides/python/nn
    # relu, relu6, crelu, elu, softplus, softsign, dropout, bias_add
    # sigmoid, tanh, leaky_relu
    activation_function = config.activation_functionDNNTF
    
    # When not None, the probability of dropout.
    dropout_perc = config.dropout_percDNNTF
    
    trainingSteps = config.trainingStepsDNNTF    # number of training steps
    
    valMonitorSecs = config.valMonitorSecsDNNTF    # perform validation every given seconds
    
    logCheckpoint = config.logCheckpointDNNTF
    timeCheckpoint = config.timeCheckpointDNNTF     # number of seconds in between checkpoints
    
    # threshold in % of probabilities for listing prediction results
    thresholdProbabilityPred = config.thresholdProbabilityPredDNNTF
    
    plotMap = config.plotMapDNNTF
    
    shuffleTrain = config.shuffleTrainDNNTF
    shuffleTest = config.shuffleTestDNNTF

    #*************************************************
    # Setup variables and definitions- do not change.
    #*************************************************
    if runDNNTF == True:
        import tensorflow as tf
        if activation_function == "sigmoid" or activation_function == "tanh":
            actFn = "tf."+activation_function
        else:
            actFn = "tf.nn."+activation_function
        activationFn = eval(actFn)

        if optimizer == "ProximalAdagrad":
            print(" DNNTF: Using ProximalAdagrad, learn_rate:",learning_rate,
                  ", l2_reg_strength:", l2_reg_strength,"\n")
            optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate,
                                        l2_regularization_strength=l2_reg_strength,
                                        use_locking=False,
                                        name="ProximalAdagrad")
        if optimizer == "AdamOpt":
            print(" DNNTF: Using Adam, learn_rate:",learning_rate,"\n")
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08,
                                        use_locking=False,
                                        name="Adam")
        if optimizer == "Adadelta":
            print(" DNNTF: Using Adadelta, learn_rate:",learning_rate,"\n")
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                        rho=0.95,
                                        epsilon=1e-08,
                                        use_locking=False,
                                        name="Adadelta")

        if optimizer == "GradientDescent":
            print(" DNNTF: Using GradientDescent, learn_rate:",learning_rate,"\n")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,
                                        use_locking=False,
                                        name="GradientDescent")

        if optimizer == "ProximalGradientDescent":
            print(" DNNTF: Using ProximalAdagrad, learn_rate:",learning_rate,
                  ", l2_reg_strength:", l2_reg_strength,"\n")
            optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate,
                                        l2_regularization_strength=l2_reg_strength,
                                        use_locking=False,
                                        name="ProximalGradientDescent")

#**********************************************
''' Deep Neural Networks - sklearn'''
#**********************************************
class nnDef:
    config = Configuration()
    config.readConfig(config.configFile)
    
    runNN = config.runNN
    
    alwaysRetrain = config.alwaysRetrainNN
    
    # Format: (number_neurons_HL1, number_neurons_HL2, number_neurons_HL3,)
    hidden_layers = config.hidden_layersNN  # default: 200
    
    # Optimizers:
    # - adam (default), for large datasets
    # - lbfgs (default) for smaller datasets
    optimizer = config.optimizerNN
    
    # activation functions: http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    # identity, logistic (sigmoid), tanh, relu
    
    activation_function = config.activation_functionNN
    
    l2_reg_strength = config.l2_reg_strengthNN
    
    MLPRegressor = config.MLPRegressorNN
    
    # threshold in % of probabilities for listing prediction results
    thresholdProbabilityPred = config.thresholdProbabilityPredNN
    
    plotMap = config.plotMapNN
    nnClassReport = config.classReportNN

#**********************************************
''' Support Vector Machines'''
#**********************************************
class svmDef:
    config = Configuration()
    config.readConfig(config.configFile)

    runSVM = config.runSVM
    
    alwaysRetrain = config.alwaysRetrainSVM
    
    # threshold in % of probabilities for listing prediction results
    thresholdProbabilitySVMPred = config.thresholdProbabilityPredSVM
    
    ''' Training algorithm for SVM
        Use either 'linear' or 'rbf'
        ('rbf' for large number of features) '''
    
    Cfactor = config.CfactorSVM
    kernel = config.kernelSVM
    showClasses = config.showClassesSVM

    plotMap = config.plotMapSVM
    svmClassReport = config.classReportSVM

#**********************************************
''' Principal component analysis (PCA) '''
#**********************************************
class pcaDef:
    config = Configuration()
    config.readConfig(config.configFile)
    
    runPCA = config.runPCA
    customNumPCAComp = config.customNumCompPCA
    numPCAcomponents = config.numComponentsPCA

#**********************************************
''' K-means '''
#**********************************************
class kmDef:
    config = Configuration()
    config.readConfig(config.configFile)

    runKM = config.runKM
    customNumKMComp = config.customNumCompKM
    numKMcomponents = config.numComponentsKM
    plotKM = config.plotKM
    plotMap = config.plotMapKM

#**********************************************
''' TensorFlow '''
#**********************************************
class tfDef:
    config = Configuration()
    config.readConfig(config.configFile)
    
    runTF = config.runTF

    alwaysRetrain = config.alwaysRetrainTF
    alwaysImprove = config.alwaysImproveTF       # alwaysRetrain must be "True" for this to work
    
    # threshold in % of probabilities for listing prediction results
    thresholdProbabilityTFPred = config.thresholdProbabilityPredTF
    
    decayLearnRate = config.decayLearnRateTF
    learnRate = config.learnRateTF
    
    subsetCrossValid = config.subsetCrossValidTF
    percentCrossValid = config.percentCrossValidTF
    
    logCheckpoint = config.logCheckpointTF

    plotMap = config.plotMapTF
    plotClassDistribTF = config.plotClassDistribTF
    enableTensorboard = config.enableTensorboardTF

#**********************************************
''' Plotting '''
#**********************************************
class plotDef:
    config = Configuration()
    config.readConfig(config.configFile)
    
    showProbPlot = config.showProbPlot
    showPCAPlots = config.showPCAPlots
    createTrainingDataPlot = config.createTrainingDataPlot
    showTrainingDataPlot = config.showTrainingDataPlot
    plotAllSpectra = config.plotAllSpectra  # Set to false for extremely large training sets
    
    if plotAllSpectra == "False":
        stepSpectraPlot = config.stepSpectraPlot  # steps in the number of spectra to be plotted

#**********************************************
''' Multiprocessing '''
#**********************************************
#multiproc = config.multiproc

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
            try:
                trainAccuracy(sys.argv[2], testFile)
            except:
                usage()
                sys.exit(2)

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
        runPCA(En, Cl, A, YnormXind, pcaDef.numPCAcomponents)

    ''' Open prediction file '''
    R, Rx = readPredFile(sampleFile)

    ''' Preprocess prediction data '''
    A, Cl, En, Aorig = preProcessNormLearningData(A, En, Cl, YnormXind, 0)
    R, Rorig = preProcessNormPredData(R, Rx, En, YnormXind, 0)
    
    ''' Run Neural Network - TensorFlow'''
    if dnntfDef.runDNNTF == True:
        dnntfDef.alwaysImprove = False
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
    R, Rorig = preProcessNormPredData(R, Rx, En,YnormXind, 0)

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
        r, rorig = preProcessNormPredData(r, Rx, En, YnormXind, type)
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

#********************************************************************************
''' TensorFlow '''
''' Run SkFlow - DNN Classifier '''
''' https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier'''
#********************************************************************************
''' Train DNNClassifier model training via TensorFlow-skflow '''
#********************************************************************************
def trainDNNTF(A, Cl, A_test, Cl_test, Root):
    print('==========================================================================\n')
    print('\033[1m Running Deep Neural Networks: skflow-DNNClassifier - TensorFlow...\033[0m')
    print('  Hidden layers:', dnntfDef.hidden_layers)
    print('  Optimizer:',dnntfDef.optimizer,
                '\n  Activation function:',dnntfDef.activation_function,
                '\n  L2:',dnntfDef.l2_reg_strength,
                '\n  Dropout:', dnntfDef.dropout_perc)
    import tensorflow as tf
    import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing
    
    if dnntfDef.logCheckpoint ==True:
        tf.logging.set_verbosity(tf.logging.INFO)
    
    if dnntfDef.alwaysRetrain == False:
        model_directory = Root + "/DNN-TF_" + str(len(dnntfDef.hidden_layers))+"HL_"+str(dnntfDef.hidden_layers[0])
        print("\n  Training model saved in: ", model_directory, "\n")
    else:
        dnntfDef.alwaysImprove = True
        model_directory = None
        print("\n  Training model not saved\n")

    #**********************************************
    ''' Initialize Estimator and training data '''
    #**********************************************
    print(' Initializing TensorFlow...')
    tf.reset_default_graph()

    totA = np.vstack((A, A_test))
    totCl = np.append(Cl, Cl_test)
    numTotClasses = np.unique(totCl).size
    
    le = preprocessing.LabelEncoder()
    totCl2 = le.fit_transform(totCl)
    Cl2 = le.transform(Cl)
    Cl2_test = le.transform(Cl_test)
    
    validation_monitor = skflow.monitors.ValidationMonitor(input_fn=lambda: input_fn(A_test, Cl2_test),
                                                           eval_steps=1,
                                                           every_n_steps=dnntfDef.valMonitorSecs)

    feature_columns = skflow.infer_real_valued_columns_from_input(totA.astype(np.float32))
    '''
    clf = skflow.DNNClassifier(feature_columns=feature_columns, hidden_units=dnntfDef.hidden_layers,
            optimizer=dnntfDef.optimizer, n_classes=numTotClasses,
            activation_fn=dnntfDef.activationFn, model_dir=model_directory,
            config=skflow.RunConfig(save_checkpoints_secs=dnntfDef.timeCheckpoint),
            dropout=dnntfDef.dropout_perc)
    '''
    clf = skflow.DNNClassifier(feature_columns=feature_columns, hidden_units=dnntfDef.hidden_layers,
            optimizer=dnntfDef.optimizer, n_classes=numTotClasses,
            activation_fn=dnntfDef.activationFn, model_dir=model_directory,
            config=tf.estimator.RunConfig().replace(save_summary_steps=dnntfDef.timeCheckpoint),
            dropout=dnntfDef.dropout_perc)
                               
    print("\n Number of global steps:",dnntfDef.trainingSteps)

    #**********************************************
    ''' Train '''
    #**********************************************
    if dnntfDef.alwaysImprove == True or os.path.exists(model_directory) is False:
        print(" (Re-)training using dataset: ", Root,"\n")
        clf.fit(input_fn=lambda: input_fn(A, Cl2),
                steps=dnntfDef.trainingSteps, monitors=[validation_monitor])
    else:
        print("  Retreaving training model from: ", model_directory,"\n")

    accuracy_score = clf.evaluate(input_fn=lambda: input_fn(A_test, Cl2_test), steps=1)
    print('\n  ===================================')
    print('  \033[1msk-DNN-TF\033[0m - Accuracy')
    print('  ===================================')
    print("\n  Accuracy: {:.2f}%".format(100*accuracy_score["accuracy"]))
    print("  Loss: {:.2f}".format(accuracy_score["loss"]))
    print("  Global step: {:.2f}\n".format(accuracy_score["global_step"]))
    print('  ===================================\n')

    return clf, le

#********************************************************************************
''' Predict using DNNClassifier model via TensorFlow-skflow '''
#********************************************************************************
def predDNNTF(clf, le, R, Cl):
    import tensorflow as tf
    import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing

    #**********************************************
    ''' Predict '''
    #**********************************************
    def input_fn_predict():
        x = tf.constant(R.astype(np.float32))
        return x

    pred_class = list(clf.predict_classes(input_fn=input_fn_predict))[0]
    predValue = le.inverse_transform(pred_class)
    prob = list(clf.predict_proba(input_fn=input_fn_predict))[0]
    predProb = round(100*prob[pred_class],2)
    
    rosterPred = np.where(prob>dnntfDef.thresholdProbabilityPred/100)[0]
    
    print('\n  ===================================')
    print('  \033[1msk-DNN-TF\033[0m - Probability >',str(dnntfDef.thresholdProbabilityPred),'%')
    print('  ===================================')
    print('  Prediction\tProbability [%]')
    for i in range(rosterPred.shape[0]):
        print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.4f}'.format(100*prob[rosterPred][i])))
    print('  ===================================')
    
    print('\033[1m' + '\n Predicted value (skflow.DNNClassifier) = ' + predValue +
          '  (probability = ' + str(predProb) + '%)\033[0m\n')

    return predValue, predProb

#**********************************************
''' Format input data for Estimator '''
#**********************************************
def input_fn(A, Cl2):
    import tensorflow as tf
    x = tf.constant(A.astype(np.float32))
    y = tf.constant(Cl2)
    return x,y

#********************************************************************************
''' TensorFlow '''
''' Run tf.estimator.DNNClassifier '''
''' https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier'''
#********************************************************************************
''' Train DNNClassifier model training via TensorFlow-Estimators '''
#********************************************************************************
def trainDNNTF2(A, Cl, A_test, Cl_test, Root):
    printInfo()
    import tensorflow as tf
    import tensorflow.contrib.learn as skflow
    from sklearn import preprocessing
    from tensorflow.contrib.learn.python.learn import monitors as monitor_lib
    
    if dnntfDef.logCheckpoint ==True:
        tf.logging.set_verbosity(tf.logging.INFO)
    
    if dnntfDef.alwaysRetrain == False:
        model_directory = Root + "/DNN-TF_" + str(len(dnntfDef.hidden_layers))+"HL_"+str(dnntfDef.hidden_layers[0])
        print("\n  Training model saved in: ", model_directory, "\n")
    else:
        dnntfDef.alwaysImprove = True
        model_directory = None
        print("\n  Training model not saved\n")

    #**********************************************
    ''' Initialize Estimator and training data '''
    #**********************************************
    print(' Initializing TensorFlow...')
    tf.reset_default_graph()

    totA = np.vstack((A, A_test))
    totCl = np.append(Cl, Cl_test)
    numTotClasses = np.unique(totCl).size
    
    le = preprocessing.LabelEncoder()
    totCl2 = le.fit_transform(totCl)
    Cl2 = le.transform(Cl)
    Cl2_test = le.transform(Cl_test)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(A)},
            y=np.array(Cl2),
            num_epochs=None,
            shuffle=dnntfDef.shuffleTrain)
        
    test_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": np.array(A_test)},
            y=np.array(Cl2_test),
            num_epochs=1,
            shuffle=dnntfDef.shuffleTest)
    
    validation_monitor = [skflow.monitors.ValidationMonitor(input_fn=test_input_fn,
                                                           eval_steps=1,
                                                           every_n_steps=dnntfDef.valMonitorSecs)]

    feature_columns = [tf.feature_column.numeric_column("x", shape=[totA.shape[1]])]
    
    clf = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=dnntfDef.hidden_layers,
            optimizer=dnntfDef.optimizer, n_classes=numTotClasses,
            activation_fn=dnntfDef.activationFn, model_dir=model_directory,
            config=tf.estimator.RunConfig().replace(save_summary_steps=dnntfDef.timeCheckpoint),
            dropout=dnntfDef.dropout_perc)
           
    hooks = monitor_lib.replace_monitors_with_hooks(validation_monitor, clf)

    #**********************************************
    ''' Define parameters for savedmodel '''
    #**********************************************
    feature_spec = {'x': tf.FixedLenFeature([numTotClasses],tf.float32)}
    def serving_input_receiver_fn():
        serialized_tf_example = tf.placeholder(dtype=tf.string,
                                         shape=[None],
                                         name='input_tensors')
        receiver_tensors = {'inputs': serialized_tf_example}
        features = tf.parse_example(serialized_tf_example, feature_spec)
        return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

    print("\n Number of global steps:",dnntfDef.trainingSteps)

    #**********************************************
    ''' Train '''
    #**********************************************
    if dnntfDef.alwaysImprove == True or os.path.exists(model_directory) is False:
        print(" (Re-)training using dataset: ", Root,"\n")
        clf.train(input_fn=train_input_fn,
                steps=dnntfDef.trainingSteps, hooks=hooks)
        print(" Exporting savedmodel in: ", Root,"\n")
        clf.export_savedmodel(model_directory, serving_input_receiver_fn)
    else:
        print("  Retreaving training model from: ", model_directory,"\n")

    accuracy_score = clf.evaluate(input_fn=test_input_fn, steps=1)
    printInfo()

    print('\n  ==================================')
    print('  \033[1mtf.DNNCl\033[0m - Accuracy')
    print('  ==================================')
    print("\n  Accuracy: {:.2f}%".format(100*accuracy_score["accuracy"]))
    print("  Loss: {:.2f}".format(accuracy_score["loss"]))
    print("  Global step: {:.2f}\n".format(accuracy_score["global_step"]))
    print('  ==================================\n')

    return clf, le

def printInfo():
    print('==========================================================================\n')
    print('\033[1m Running Deep Neural Networks: tf.DNNClassifier - TensorFlow...\033[0m')
    print('  Optimizer:',dnntfDef.optimizer,
          '\n  Hidden layers:', dnntfDef.hidden_layers,
          '\n  Activation function:',dnntfDef.activation_function,
          '\n  L2:',dnntfDef.l2_reg_strength,
          '\n  Dropout:', dnntfDef.dropout_perc,
          '\n  Learning rate:', dnntfDef.learning_rate,
          '\n  Shuffle Train:', dnntfDef.shuffleTrain,
          '\n  Shuffle Test:', dnntfDef.shuffleTest,)

#********************************************************************************
''' Predict using tf.estimator.DNNClassifier model via TensorFlow '''
#********************************************************************************
def predDNNTF2(clf, le, R, Cl):
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

#********************************************************************************
''' Support Vector Machines - SVM '''
''' http://scikit-learn.org/stable/modules/svm.html '''
#********************************************************************************
''' Train SVM '''
#********************************************************************************
def trainSVM(A, Cl, A_test, Cl_test, Root):
    from sklearn import svm
    from sklearn.externals import joblib
    svmTrainedData = Root + '.svmModel.pkl'
    print('==========================================================================\n')
    print('\033[1m Running Support Vector Machine (kernel: ' + svmDef.kernel + ')\033[0m')
    try:
        if svmDef.alwaysRetrain == False:
            with open(svmTrainedData):
                print('  Opening SVM training model...\n')
                clf = joblib.load(svmTrainedData)
        else:
            raise ValueError('  Force retraining SVM model')
    except:
        #**********************************************
        ''' Retrain training model if not available'''
        #**********************************************
        print('  Retraining SVM data...')
        clf = svm.SVC(C = svmDef.Cfactor, decision_function_shape = 'ovr', probability=True)
        
        print("  Training on the full training dataset\n")
        clf.fit(A,Cl)
        accur = clf.score(A_test,Cl_test)
        print('  Mean accuracy: ',100*accur,'%')

        Z = clf.decision_function(A)
        print('\n  Number of classes = ' + str(Z.shape[1]))
        joblib.dump(clf, svmTrainedData)
        if svmDef.showClasses == True:
            print('  List of classes: ' + str(clf.classes_))

    print('\n==========================================================================\n')
    return clf

#********************************************************************************
''' Predict using SVM '''
#********************************************************************************
def predSVM(clf, A, Cl, R):
    R_pred = clf.predict(R)
    prob = clf.predict_proba(R)[0].tolist()
    
    rosterPred = np.where(clf.predict_proba(R)[0]>svmDef.thresholdProbabilitySVMPred/100)[0]
    print('\n  ==============================')
    print('  \033[1mSVM\033[0m - Probability >',str(svmDef.thresholdProbabilitySVMPred),'%')
    print('  ==============================')
    print('  Prediction\tProbability [%]')
    for i in range(rosterPred.shape[0]):
        print(' ',str(np.unique(Cl)[rosterPred][i]),'\t\t',str('{:.1f}'.format(100*clf.predict_proba(R)[0][rosterPred][i])))
    print('  ==============================')

    print('\033[1m' + '\n Predicted value (SVM) = ' + str(R_pred[0]) + ' (probability = ' +
      str(round(100*max(prob),1)) + '%)\033[0m\n')
    
    #**************************************
    ''' SVM Classification Report '''
    #**************************************
    if svmDef.svmClassReport == True:
        print(' SVM Classification Report \n')
        runClassReport(clf, A, Cl)

    #*************************
    ''' Plot probabilities '''
    #*************************
    if plotDef.showProbPlot == True:
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
def runPCA(En, Cl, A, YnormXind, numPCAcomponents):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt
    from matplotlib import cm

    #''' Open and process training data '''
    #En, Cl, A, YnormXind = readLearnFile(learnFile)

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

    if plotDef.showPCAPlots == True:
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
    
    '''
    for i in range(0, numKMcomp):
        print('\n Class: ' + str(i) + '\n  ',end="")
        for j in range(0,kmeans.labels_.shape[0]):
            if kmeans.labels_[j] == i:
                print(' ' + str(Cl[j]), end="")
    '''
    print('\n  ==============================')
    print('  \033[1mKM\033[0m - Predicted class: \033[1m',str(kmeans.predict(R)[0]),'\033[0m')
    print('  ==============================')
    print('  Prediction')
    for j in range(0,kmeans.labels_.shape[0]):
        if kmeans.labels_[j] == 22:
            print('  ' + str(Cl[j]))
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

#************************************
''' Read Learning file '''
#************************************
def readLearnFile(learnFile):
    try:
        with open(learnFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Learning file not found \n' + '\033[0m')
        return

    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(M,np.s_[0:1],0)
    Cl = ['{:.2f}'.format(x) for x in M[:,0]]
    A = np.delete(M,np.s_[0:1],1)
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

    print(' Number of datapoints = ' + str(A.shape[0]))
    print(' Size of each datapoint = ' + str(A.shape[1]) + '\n')
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
def preProcessNormPredData(R, Rx, En,YnormXind, type):
    print(' Processing Prediction data file... ')
    #**********************************************************************************
    ''' Reformat x-axis in case it does not match that of the training data '''
    #**********************************************************************************
    if(R.shape[0] != En.shape):
        if type == 0:
            print('\033[1m' + '  WARNING: Different number of datapoints for the x-axis\n  for training (' + str(En.shape[1]) + ') and sample (' + str(R.shape[0]) + ') data.\n  Reformatting x-axis of sample data...\n' + '\033[0m')
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
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print(' Single files:')
    print('  python3 SpectraLearnPredict.py -f <learningfile> <spectrafile>\n')
    print(' Cross-validation for accuracy determination:')
    print('  python3 SpectraLearnPredict.py -a <learningfile> <testdataset>\n')
    print(' Cross-validation for accuracy determination (automatic splitting):')
    print('  python3 SpectraLearnPredict.py -a <learningfile>\n')
    print(' Maps (formatted for Horiba LabSpec):')
    print('  python3 SpectraLearnPredict.py -m <learningfile> <spectramap>\n')
    print(' Batch txt files:')
    print('  python3 SpectraLearnPredict.py -b <learningfile>\n')
    print(' K-means on maps:')
    print('  python3 SpectraLearnPredict.py -k <spectramap> <number_of_classes>\n')
    print(' Principal component analysis on spectral collection files: ')
    print('  python3 SpectraLearnPredict.py -p <spectrafile> <#comp>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')


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

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
    
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

        save_path = saver.save(sess, tfTrainedData)
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

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
