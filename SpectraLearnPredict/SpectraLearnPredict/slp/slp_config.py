#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict - CONFIG
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
        
        self.Ynorm = eval(self.preprocDef['Ynorm'])
        self.fullYnorm = eval(self.preprocDef['fullYnorm'])
        self.StandardScalerFlag = eval(self.preprocDef['StandardScalerFlag'])
        self.subsetCrossValid = eval(self.preprocDef['subsetCrossValid'])
        self.percentCrossValid = eval(self.preprocDef['percentCrossValid'])
        self.YnormTo = eval(self.preprocDef['YnormTo'])
        self.YnormX = eval(self.preprocDef['YnormX'])
        self.YnormXdelta = eval(self.preprocDef['YnormXdelta'])
        self.enRestrictRegion = eval(self.preprocDef['enRestrictRegion'])
        self.enLim1 = eval(self.preprocDef['enLim1'])
        self.enLim2 = eval(self.preprocDef['enLim2'])
        self.scrambleNoiseFlag = eval(self.preprocDef['scrambleNoiseFlag'])
        self.scrambleNoiseOffset = eval(self.preprocDef['scrambleNoiseOffset'])
        self.cherryPickEnPoint = eval(self.preprocDef['cherryPickEnPoint'])
        self.enSel = eval(self.preprocDef['enSel'])
        self.enSelDelta = eval(self.preprocDef['enSelDelta'])
        
        self.runDNNTF = eval(self.dnntfDef['runDNNTF'])
        self.runSkflowDNNTF = eval(self.dnntfDef['runSkflowDNNTF'])
        self.alwaysRetrainDNNTF = eval(self.dnntfDef['alwaysRetrainDNNTF'])
        self.alwaysImproveDNNTF = eval(self.dnntfDef['alwaysImproveDNNTF'])
        self.hidden_layersDNNTF = eval(self.dnntfDef['hidden_layersDNNTF'])
        self.optimizerDNNTF = self.dnntfDef['optimizerDNNTF']
        self.learning_rateDNNTF = eval(self.dnntfDef['learning_rateDNNTF'])
        self.l2_reg_strengthDNNTF = eval(self.dnntfDef['l2_reg_strengthDNNTF'])
        self.activation_functionDNNTF = self.dnntfDef['activation_functionDNNTF']
        self.dropout_percDNNTF = eval(self.dnntfDef['dropout_percDNNTF'])
        self.trainingStepsDNNTF = eval(self.dnntfDef['trainingStepsDNNTF'])
        self.valMonitorSecsDNNTF = eval(self.dnntfDef['valMonitorSecsDNNTF'])
        self.logCheckpointDNNTF = eval(self.dnntfDef['logCheckpointDNNTF'])
        self.timeCheckpointDNNTF = eval(self.dnntfDef['timeCheckpointDNNTF'])
        self.thresholdProbabilityPredDNNTF = eval(self.dnntfDef['thresholdProbabilityPredDNNTF'])
        self.plotMapDNNTF = eval(self.dnntfDef['plotMapDNNTF'])
        try:
            self.shuffleTrainDNNTF = eval(self.dnntfDef['shuffleTrainDNNTF'])
            self.shuffleTestDNNTF = eval(self.dnntfDef['shuffleTestDNNTF'])
        except:
            self.shuffleTrainDNNTF = True
            self.shuffleTestDNNTF = False
        
        self.runNN = eval(self.nnDef['runNN'])
        self.alwaysRetrainNN = eval(self.nnDef['alwaysRetrainNN'])
        self.hidden_layersNN = eval(self.nnDef['hidden_layersNN'])
        self.optimizerNN = self.nnDef['optimizerNN']
        self.activation_functionNN = self.nnDef['activation_functionNN']
        self.l2_reg_strengthNN = eval(self.nnDef['l2_reg_strengthNN'])
        self.MLPRegressorNN = eval(self.nnDef['MLPRegressorNN'])
        self.thresholdProbabilityPredNN = eval(self.nnDef['thresholdProbabilityPredNN'])
        self.plotMapNN = eval(self.nnDef['plotMapNN'])
        self.classReportNN = eval(self.nnDef['classReportNN'])
    
        self.runSVM = eval(self.svmDef['runSVM'])
        self.alwaysRetrainSVM = eval(self.svmDef['alwaysRetrainSVM'])
        self.thresholdProbabilityPredSVM = eval(self.svmDef['thresholdProbabilityPredSVM'])
        self.CfactorSVM = eval(self.svmDef['CfactorSVM'])
        self.kernelSVM = self.svmDef['kernelSVM']
        self.showClassesSVM = eval(self.svmDef['showClassesSVM'])
        self.plotMapSVM = eval(self.svmDef['plotMapSVM'])
        self.classReportSVM = eval(self.svmDef['classReportSVM'])
        
        self.runPCA = eval(self.pcaDef['runPCA'])
        self.customNumCompPCA = eval(self.pcaDef['customNumCompPCA'])
        self.numComponentsPCA = eval(self.pcaDef['numComponentsPCA'])

        self.runKM = eval(self.kmDef['runKM'])
        self.customNumCompKM = eval(self.kmDef['customNumCompKM'])
        self.numComponentsKM = eval(self.kmDef['numComponentsKM'])
        self.plotKM = eval(self.kmDef['plotKM'])
        self.plotMapKM = eval(self.kmDef['plotMapKM'])
        
        self.runTF = eval(self.tfDef['runTF'])
        self.alwaysRetrainTF = eval(self.tfDef['alwaysRetrainTF'])
        self.alwaysImproveTF = eval(self.tfDef['alwaysImproveTF'])
        self.thresholdProbabilityPredTF = eval(self.tfDef['thresholdProbabilityPredTF'])
        self.decayLearnRateTF = eval(self.tfDef['decayLearnRateTF'])
        self.learnRateTF = eval(self.tfDef['learnRateTF'])
        self.subsetCrossValidTF = eval(self.tfDef['subsetCrossValidTF'])
        self.percentCrossValidTF = eval(self.tfDef['percentCrossValidTF'])
        self.logCheckpointTF = eval(self.tfDef['logCheckpointTF'])
        self.plotMapTF = eval(self.tfDef['plotMapTF'])
        self.plotClassDistribTF = eval(self.tfDef['plotClassDistribTF'])
        self.enableTensorboardTF = eval(self.tfDef['enableTensorboardTF'])
         
        self.showProbPlot = eval(self.plotDef['showProbPlot'])
        self.showPCAPlots = eval(self.plotDef['showPCAPlots'])
        self.createTrainingDataPlot = eval(self.plotDef['createTrainingDataPlot'])
        self.showTrainingDataPlot = eval(self.plotDef['showTrainingDataPlot'])
        self.plotAllSpectra = eval(self.plotDef['plotAllSpectra'])
        self.stepSpectraPlot = eval(self.plotDef['stepSpectraPlot'])

        self.multiproc = eval(self.sysDef['multiproc'])

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
    # sigmoid, tanh
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
    
    shuffleTrainDNNTF = config.shuffleTrainDNNTF
    shuffleTestDNNTF = config.shuffleTestDNNTF

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


