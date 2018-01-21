#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict2 - CONFIG
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
        confFileName = "SpectraLearnPredict2.ini"
        self.configFile = os.getcwd()+"/"+confFileName
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        if os.path.isfile(self.configFile) is False:
            print("Configuration file: \""+confFileName+"\" does not exist: Creating one.")
            self.createConfig()

    # Hadrcoded default definitions for the confoguration file
    def preprocDef(self):
        self.conf['Preprocessing'] = {
            'Ynorm' : True,
            'fullYnorm' : True,
            'StandardScalerFlag' : False,
            'subsetCrossValid' : False,
            'percentCrossValid' : 0.01,
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
            'runSkflowDNNTF' : False,
            'alwaysRetrainDNNTF' : False,
            'alwaysImproveDNNTF' : True,
            'hidden_layersDNNTF' : [400,],
            'optimizerDNNTF' : "ProximalAdagrad",
            'learning_rateDNNTF' : 0.1,
            'learning_rate_decayDNNTF' : False,
            'learning_rate_decay_rateDNNTF' : 0.96,
            'learning_rate_decay_stepsDNNTF' : 100,
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

    def kerasDef(self):
        self.conf['Keras'] = {
            'runKeras' : True,
            'alwaysRetrainDNNTF' : False,
            'alwaysImproveDNNTF' : True,
            'hidden_layersKeras' : [400,400],
            'optimizerKeras' : "SGD",
            'l2_reg_strengthKeras' : 1e-4,
            'learning_rateKeras' : 1e-4,
            'learning_decay_rateKeras' : 1e-6,
            'activation_functionKeras' : "relu",
            'dropout_percKeras' : 0.5,
            'trainingStepsKeras' : 1000,
            'thresholdProbabilityPredKeras' : 0.01,
            'plotModelKeras' : False,
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
            'multiProc' : False,
            'useAllCores' : False,
            'numCores' : 2,
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
        self.kerasDef = self.conf['Keras']
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
        self.learning_rate_decayDNNTF = self.conf.getboolean('DNNClassifier','learning_rate_decayDNNTF')
        self.learning_rate_decay_rateDNNTF = self.conf.getfloat('DNNClassifier','learning_rate_decay_rateDNNTF')
        self.learning_rate_decay_stepsDNNTF = self.conf.getfloat('DNNClassifier','learning_rate_decay_stepsDNNTF')
        self.l2_reg_strengthDNNTF = self.conf.getfloat('DNNClassifier','l2_reg_strengthDNNTF')
        self.activation_functionDNNTF = self.dnntfDef['activation_functionDNNTF']
        self.dropout_percDNNTF = eval(self.dnntfDef['dropout_percDNNTF'])
        self.trainingStepsDNNTF = self.conf.getint('DNNClassifier','trainingStepsDNNTF')
        self.valMonitorSecsDNNTF = self.conf.getint('DNNClassifier','valMonitorSecsDNNTF')
        self.logCheckpointDNNTF = self.conf.getboolean('DNNClassifier','logCheckpointDNNTF')
        self.timeCheckpointDNNTF = self.conf.getint('DNNClassifier','timeCheckpointDNNTF')
        self.thresholdProbabilityPredDNNTF = self.conf.getfloat('DNNClassifier','thresholdProbabilityPredDNNTF')
        self.plotMapDNNTF = self.conf.getboolean('DNNClassifier','plotMapDNNTF')
        self.shuffleTrainDNNTF = self.conf.getboolean('DNNClassifier','shuffleTrainDNNTF')
        self.shuffleTestDNNTF = self.conf.getboolean('DNNClassifier','shuffleTestDNNTF')
        
        self.runKeras = self.conf.getboolean('Keras','runKeras')
        self.alwaysRetrainKeras = self.conf.getboolean('Keras','alwaysRetrainKeras')
        self.alwaysImproveKeras = self.conf.getboolean('Keras','alwaysImproveKeras')
        self.hidden_layersKeras = eval(self.kerasDef['hidden_layersKeras'])
        self.optimizerKeras = self.kerasDef['optimizerKeras']
        self.l2_reg_strengthKeras = self.conf.getfloat('Keras','l2_reg_strengthKeras')
        self.learning_rateKeras = self.conf.getfloat('Keras','learning_rateKeras')
        self.learning_decay_rateKeras = self.conf.getfloat('Keras','learning_decay_rateKeras')
        self.activation_functionKeras = self.kerasDef['activation_functionKeras']
        self.dropout_percKeras = eval(self.kerasDef['dropout_percKeras'])
        self.trainingStepsKeras = self.conf.getint('Keras','trainingStepsKeras')
        self.thresholdProbabilityPredKeras = self.conf.getfloat('Keras','thresholdProbabilityPredKeras')
        self.plotModelKeras = self.conf.getboolean('Keras','plotModelKeras')
        
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

        self.multiProc = self.conf.getboolean('System','multiProc')
        self.useAllCores = self.conf.getboolean('System','useAllCores')
        self.numCores = self.conf.getint('System','numCores')

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
            self.kerasDef()
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
    #                        GradientDescent, ProximalGradientDescent,
    # https://www.tensorflow.org/api_guides/python/train
    optimizer = config.optimizerDNNTF
    
    learning_rate = config.learning_rateDNNTF
    learning_rate_decay = config.learning_rate_decayDNNTF
    if learning_rate_decay == True:
        learning_rate_decay_rate = config.learning_rate_decay_rateDNNTF
        learning_rate_decay_steps = config.learning_rate_decay_stepsDNNTF

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
            optimizer_tag = " ProximalAdagrad, learn_rate: "+str(learning_rate)+\
                  ", l2_reg_strength: "+str(l2_reg_strength)
            optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate,
                                        l2_regularization_strength=l2_reg_strength,
                                        use_locking=False,
                                        name="ProximalAdagrad")
        if optimizer == "AdamOpt":
            optimizer_tag = " Adam, learn_rate: "+str(learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08,
                                        use_locking=False,
                                        name="Adam")
        if optimizer == "Adadelta":
            optimizer_tag = " Adadelta, learn_rate: "+str(learning_rate)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,
                                        rho=0.95,
                                        epsilon=1e-08,
                                        use_locking=False,
                                        name="Adadelta")

        if optimizer == "GradientDescent":
            optimizer_tag = " GradientDescent, learn_rate: "+str(learning_rate)
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate,
                                        use_locking=False,
                                        name="GradientDescent")

        if optimizer == "ProximalGradientDescent":
            optimizer_tag = " ProximalAdagrad, learn_rate: "+str(learning_rate)+\
                  ", l2_reg_strength: "+str(l2_reg_strength)
            optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate,
                                        l2_regularization_strength=l2_reg_strength,
                                        use_locking=False,
                                        name="ProximalGradientDescent")

#***********************************************************
''' Deep Neural Networks - Keras'''
#***********************************************************
class kerasDef:
    config = Configuration()
    config.readConfig(config.configFile)
    
    runKeras = config.runKeras
    alwaysRetrain = config.alwaysRetrainKeras
    alwaysImprove = config.alwaysImproveKeras
    
    # Format: [number_neurons_HL1, number_neurons_HL2, number_neurons_HL3,...]
    hidden_layers = config.hidden_layersKeras

    # Stock Optimizers: Adagrad (recommended), Adam, Ftrl, RMSProp, SGD
    # https://www.tensorflow.org/api_guides/python/train
    #optimizer = "Adagrad"

    # Additional optimizers: ProximalAdagrad, AdamOpt, Adadelta,
    #                        GradientDescent, ProximalGradientDescent,
    # https://www.tensorflow.org/api_guides/python/train
    optimizer = config.optimizerKeras
    
    l2_reg_strength = config.l2_reg_strengthKeras
    
    learning_rate = config.learning_rateKeras
    learning_decay_rate = config.learning_decay_rateKeras
    
    
    # activation functions: https://keras.io/activations/
    # softmax, elu, relu, selu, softplus, softsign, tanh, sigmoid,
    # hard_sigmoid, linear
    activation_function = config.activation_functionKeras
    
    # When not None, the probability of dropout.
    dropout_perc = config.dropout_percKeras
    
    trainingSteps = config.trainingStepsKeras    # number of training steps
    thresholdProbabilityPred = config.thresholdProbabilityPredKeras
    
    plotModel = config.plotModelKeras

    #*************************************************
    # Setup variables and definitions- do not change.
    #*************************************************
    if runKeras == True:
        import tensorflow as tf
        import keras.optimizers as opt
        from keras.layers import Activation
            
        if optimizer == "SGD":
            optimizer_tag = " SGD, learn_rate: "+str(learning_rate)
            optimizer = opt.SGD(lr=learning_rate, decay=learning_decay_rate,
                momentum=0.9, nesterov=True)
            
        if optimizer == "Adagrad":
            optimizer_tag = " Adagrad, learn_rate: "+str(learning_rate)
            optimizer = opt.Adagrad(lr=learning_rate, epsilon=1e-08,
                decay=learning_decay_rate)
        
        if optimizer == "Adadelta":
            optimizer_tag = " AdaDelta, learn_rate: "+str(learning_rate)
            optimizer = opt.Adadelta(lr=learning_rate, epsilon=1e-08, rho=0.95,
                decay=learning_decay_rate)
            
        if optimizer == "Adam":
            optimizer_tag = " Adam, learn_rate: "+str(learning_rate)
            optimizer = opt.Adam(lr=learning_rate, beta_1=0.9,
                                        beta_2=0.999, epsilon=1e-08,
                                        decay=learning_decay_rate,
                                        amsgrad=False)

        if optimizer == "Adamax":
            optimizer_tag = " Adamax, learn_rate: "+str(learning_rate)
            optimizer = opt.Adamax(lr=learning_rate, beta_1=0.9,
                                        beta_2=0.999, epsilon=1e-08,
                                        decay=learning_decay_rate)

        if optimizer == "RMSprop":
            optimizer_tag = " RMSprop, learn_rate: "+str(learning_rate)
            optimizer = opt.RMSprop(lr=learning_rate, rho=0.95,
                                        epsilon=1e-08,
                                        decay=learning_decay_rate)

        '''
        if optimizer == "TFOptimizer":
            print(" DNNTF: Using TensorFlow native optimizer"\n")
            optimizer = TFOptimizer(optimizer)
        '''

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
''' System '''
#**********************************************
class sysDef:
    config = Configuration()
    config.readConfig(config.configFile)

    multiProc = config.multiProc
    config.useAllCores
    import multiprocessing as mp
    if config.useAllCores == True:
        numCores = mp.cpu_count()
        print("\n Multiprocessing batch using max number of cores/processors: ", numCores,"\n")
    else:
        numCores = config.numCores
        print("\n Multiprocessing batch using",numCores, "cores/processors\n")

