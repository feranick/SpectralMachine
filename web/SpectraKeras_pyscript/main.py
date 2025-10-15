#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************
* SpectraKeras_CNN Classifier and Regressor
* Simplified web version
* v2025.10.15.1
* Uses: TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
**********************************************
'''
#print(__doc__)

import numpy as np
import pandas as pd
from io import BytesIO
import sys, configparser, ast, io, csv, os.path, glob, platform
from js import document, Blob, URL, console, window
from pyscript import fetch, document
import _pickle as pickle
from libSpectraKeras import *

window.py_getMineral = getMineral

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self, configIni):

        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str

        self.readConfig(configIni)
        self.model_directory = "/"
        if self.regressor:
            self.modelName = "model_regressor_CNN.h5"
            self.summaryFileName = "summary_regressor_CNN.csv"
            self.model_png = self.model_directory+"model_regressor_CNN.png"
        else:
            self.predProbThreshold = 90.00
            self.modelName = "model_classifier_CNN.h5"
            self.summaryFileName = "summary_classifier_CNN.csv"
            self.summaryAccFileName = "summary_classifier_CNN_accuracy.csv"
            self.model_png = self.model_directory+"model_classifier_CNN.png"

        self.tb_directory = "model_CNN"
        self.model_name = self.model_directory+self.modelName
        
        if self.kerasVersion == 3:
            self.model_name = os.path.splitext(self.model_name)[0]+".keras"
        
        self.model_le = self.model_directory+"model_le.pkl"
        self.spectral_range = "model_spectral_range.pkl"
        self.table_names = self.model_directory+"AAA_table_names.h5"

        self.actPlotTrain = self.model_directory+"model_CNN_train-activations_conv2D_"
        self.actPlotPredict = self.model_directory+"model_CNN_pred-activations_"
        self.sizeColPlot = 4

        if platform.system() == 'Linux':
            self.edgeTPUSharedLib = "libedgetpu.so.1"
        if platform.system() == 'Darwin':
            self.edgeTPUSharedLib = "libedgetpu.1.dylib"
        if platform.system() == 'Windows':
            self.edgeTPUSharedLib = "edgetpu.dll"
        
    def SKDef(self):
        self.conf['Parameters'] = {
            'regressor' : False,
            'normalize' : True,
            'l_rate' : 0.001,
            'l_rdecay' : 0.96,
            'CL_filter' : [1],
            'CL_size' : [10],
            'max_pooling' : [20],
            'dropCNN' : [0],
            'HL' : [40,70],
            'dropFCL' : 0,
            'l2' : 1e-4,
            'epochs' : 100,
            'cv_split' : 0.01,
            'fullSizeBatch' : False,
            'batch_size' : 64,
            'numLabels' : 1,
            'plotWeightsFlag' : False,
            'plotActivations' : False,
            'showValidPred' : False,
            'stopAtBest' : False,
            'saveBestModel' : False,
            'metricBestModelR' : 'val_mae',
            'metricBestModelC' : 'val_accuracy',
            }

    def sysDef(self):
        self.conf['System'] = {
            'kerasVersion' : 2,
            'makeQuantizedTFlite' : True,
            'useTFlitePred' : False,
            'TFliteRuntime' : False,
            'runCoralEdge' : False,
            }

    def readConfig(self,configFile):
        try:
            self.conf.read_string(configFile)
            self.SKPar = self.conf['Parameters']
            self.sysPar = self.conf['System']

            self.regressor = self.conf.getboolean('Parameters','regressor')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.CL_filter = ast.literal_eval(self.SKPar['CL_filter'])
            self.CL_size = ast.literal_eval(self.SKPar['CL_size'])
            self.max_pooling = ast.literal_eval(self.SKPar['max_pooling'])
            self.dropCNN = ast.literal_eval(self.SKPar['dropCNN'])
            self.HL = ast.literal_eval(self.SKPar['HL'])
            self.dropFCL = self.conf.getfloat('Parameters','dropFCL')
            self.l2 = self.conf.getfloat('Parameters','l2')
            self.epochs = self.conf.getint('Parameters','epochs')
            self.cv_split = self.conf.getfloat('Parameters','cv_split')
            self.fullSizeBatch = self.conf.getboolean('Parameters','fullSizeBatch')
            self.batch_size = self.conf.getint('Parameters','batch_size')
            self.numLabels = self.conf.getint('Parameters','numLabels')
            self.plotWeightsFlag = self.conf.getboolean('Parameters','plotWeightsFlag')
            self.plotActivations = self.conf.getboolean('Parameters','plotActivations')
            self.showValidPred = self.conf.getboolean('Parameters','showValidPred')
            self.stopAtBest = self.conf.getboolean('Parameters','stopAtBest')
            self.saveBestModel = self.conf.getboolean('Parameters','saveBestModel')
            self.metricBestModelR = self.conf.get('Parameters','metricBestModelR')
            self.metricBestModelC = self.conf.get('Parameters','metricBestModelC')
                
            self.kerasVersion = self.conf.getint('System','kerasVersion')
            self.makeQuantizedTFlite = self.conf.getboolean('System','makeQuantizedTFlite')
            self.useTFlitePred = self.conf.getboolean('System','useTFlitePred')
            self.TFliteRuntime = self.conf.getboolean('System','TFliteRuntime')
            self.runCoralEdge = self.conf.getboolean('System','runCoralEdge')
        except:
            print(" Error in reading configuration file. Please check it\n")

    # Create configuration file
    def createConfig(self):
        try:
            self.SKDef()
            self.sysDef()
            with open(self.configFile, 'w') as configfile:
                self.conf.write(configfile)
        except:
            print("Error in creating configuration file")

#************************************
# Get Data from Files
#************************************
async def getFile(folder, file, bin):
    url = "./"+folder+"/"+file
    if bin:
        data = await fetch(url).bytearray()
    else:
        data = await fetch(url).text()
    return data
    
async def getModelFolder():
    model = document.querySelector("#model").value
    if model == "Raman Spectroscopy":
        folder = "ml-raman"
    elif model == "Powder X-ray Diffraction (XRD)":
        folder = "ml-xrd"
    return folder
    
async def getSavedLibraries():
    folder = await getModelFolder()
    ini = await getFile(folder, "SpectraKeras_CNN.ini", False)
    dP = Conf(ini)
    model_le = await getFile(folder, dP.model_le, True)
    spectral_range = await getFile(folder, dP.spectral_range, True)
    EnN = pickle.loads(spectral_range)
    table_names = await getFile(folder, "AAA_table_names.csv", False)
    names = loadMineralJS(table_names)
    return dP,folder,EnN,spectral_range, names

async def getSpectraFiles(event):
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Please wait..."
    
    dP,folder,EnN,spectral_range,names  = await getSavedLibraries()
    
    inputFile = document.getElementById("inputFile").files.item(0)
    array_buf = await inputFile.arrayBuffer()
    file_bytes = array_buf.to_bytes()
    data = BytesIO(file_bytes)
    
    Rtot = np.loadtxt(data, unpack =True)
    #print("Rtot:",Rtot)
    print("Rtot.shape:",Rtot.shape)
    R = preProcess(Rtot, EnN, dP)
    #print("R1:",R)
    print("R1.shape",R.shape)
    Rcnn = formatForCNN(R)
    #print("R2:",R)
    #print("R2.shape",R.shape)
       
    window.__setattr__("pyscriptData",R)
    window.__setattr__("folder",folder)
    window.__setattr__("input_shape",Rcnn.shape)
    
    mineral = getMineralJS(names, 123)
    print(mineral)
    
    from js import loadModel, showData
    showData()
    loadModel()

#************************************
# Prediction
#************************************
def predict(testFile):
    dP = Conf()
    model = loadModel(dP)
    with open(dP.spectral_range, "rb") as f:
        EnN = pickle.load(f)

    R, good = readTestFile(testFile, EnN, dP)
    if not good:
        return
    R = formatForCNN(R)

    if dP.regressor:
        #predictions = model.predict(R).flatten()[0]
        predictions, _ = getPredictions(R, model, dP)
        print('\n  ========================================================')
        print('  CNN - Regressor - Prediction')
        print('  ========================================================')
        predValue = predictions.flatten()[0]
        print('\n  Predicted value (normalized) = {0:.2f}'.format(predValue))
        print('  ========================================================\n')

    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
    
        #predictions = model.predict(R, verbose=0)
        predictions, _ = getPredictions(R, model,dP)
        pred_class = np.argmax(predictions)
        if dP.useTFlitePred:
            predProb = round(100*predictions[0][pred_class]/255,2)
        else:
            predProb = round(100*predictions[0][pred_class],2)
        rosterPred = np.where(predictions[0]>0.1)[0]
        print('\n  ========================================================')
        print('  CNN - Classifier - Prediction')
        print('  ========================================================')

        if dP.numLabels == 1:
            if pred_class.size >0:
                predValue = le.inverse_transform(pred_class)[0]
            else:
                predValue = 0
            print('  Prediction\t| Class \t| Probability [%]')
            print('  -------------------------------------------------------- ')
            for i in range(len(predictions[0])-1):
                if predictions[0][i]>0.01:
                    if dP.useTFlitePred:
                        print("  {0:s}\t| {1:d}\t\t| {2:.2f}".format(getMineral(dP.table_names, int(le.inverse_transform(i)[0])),
                            int(le.inverse_transform(i)[0]), 100*predictions[0][i]/255))
                    else:
                        print("  {0:s}\t| {1:d}\t\t| {2:.2f}".format(getMineral(dP.table_names, int(le.inverse_transform(i)[0])),
                            int(le.inverse_transform(i)[0]), 100*predictions[0][i]))
                    
            print('\n  {0:s} (Class: {1:d}, probability = {2:.2f}%)\n'.format(getMineral(dP.table_names, int(predValue)), int(predValue), predProb))
            print('  ========================================================\n')

        else:
            pass
            '''
            print('\n ==========================================')
            print('\n Predicted value (probability = ' + str(predProb) + '%)')
            print(' ==========================================\n')
            print("  1:", str(predValue[0]),"%")
            print("  2:",str(predValue[1]),"%")
            print("  3:",str((predValue[1]/0.5)*(100-99.2-.3)),"%\n")
            print(' ==========================================\n')
            '''

    if dP.plotActivations and not dP.useTFlitePred:
        plotActivationsPredictions(R,model)

#************************************
# Batch Prediction
#************************************
def batchPredict(array, names):
    import json
    dP = Conf()
    model = loadModel(dP)
    with open(dP.spectral_range, "rb") as f:
        EnN = pickle.load(f)

    files = array.replace('\\','').replace('[','').replace(']','').split(',')
    fileName = names.replace('[','').replace(']','').split(',')

    predictions = np.zeros((0,0))
    predictions_list = []
    for file in files:
        R, good = readTestFile(file, EnN, dP)
        if good:
            R = formatForCNN(R)
            try:
                predictions_list.append(getPredictions(R, model, dP)[0].flatten())
            except:
                predictions_list = [np.array([getPredictions(R, model,dP)[0].flatten()])]
    predictions = np.array(predictions_list)

    if dP.regressor:
        print('\n  ========================================================')
        print('  CNN - Regressor - Prediction')
        print('  ========================================================')
        for i in range(predictions.shape[0]):
            predValue = predictions[i][0]
            print('  {0:s}:\n   Predicted value = {1:.2f}\n'.format(fileName[i],predValue))
        print('  ========================================================\n')

    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        
        print('\n  ========================================================')
        print('  CNN - Classifier - Prediction')
        print('  ========================================================')
        indPredProb = 0
        for i in range(predictions.shape[0]):
            pred_class = np.argmax(predictions[i])
            if dP.useTFlitePred:
                predProb = round(100*predictions[i][pred_class]/255,2)
            else:
                predProb = round(100*predictions[i][pred_class],2)
            rosterPred = np.atleast_1d(predictions[i][0]).nonzero()[0]

            if pred_class.size >0:
                predValue = le.inverse_transform(pred_class)[0]
                print('  {0:s}:\n   {1:s} (Class: {2:d}, probability = {3:.2f}%)\n'.format(fileName[i],getMineral(dP.table_names, int(predValue)), int(predValue), predProb))
            else:
                predValue = 0
                print('  {0:s}:\n   No predicted value (probability = {1:.2f}%)\n'.format(fileName[i],predProb))
            if predProb > dP.predProbThreshold:
                indPredProb += 1
        print('  ========================================================\n')
        print(" Predictions with probability > {0:.2f}:  {1:.2f}%\n".format(dP.predProbThreshold, indPredProb*100/predictions.shape[0]))

#************************************
# Accuracy determination
#************************************
def accDeterm(testFile):
    pass

#****************************************************
# Convert model to quantized TFlite
#****************************************************
def convertTflite(learnFile):
    pass
    
#****************************************************
# Format data for CNN
#****************************************************
def formatForCNN(A):
    listmatrix = []
    for i in range(A.shape[0]):
        spectra = np.dstack([A[i]])
        listmatrix.append(spectra)
    x = np.stack(listmatrix, axis=0)
    return x

#************************************
# Print NN Info
#************************************
def printParam():
    dP = Conf()
    print('\n  ================================================')
    print('  \033[1m CNN\033[0m - Parameters')
    print('  ================================================')
    print('  Optimizer:','Adam',
                '\n  Convolutional layers:', dP.CL_filter,
                '\n  Convolutional layers size:', dP.CL_size,
                '\n  Max Pooling:', dP.max_pooling,
                '\n  Dropout CNN:', dP.dropCNN,
                '\n  Hidden layers:', dP.HL,
                '\n  Activation function:','relu',
                '\n  L2:',dP.l2,
                '\n  Dropout HL:', dP.dropFCL,
                '\n  Learning rate:', dP.l_rate,
                '\n  Learning decay rate:', dP.l_rdecay)
    if dP.fullSizeBatch:
        print('  Batch size: full')
    else:
        print('  Batch size:', dP.batch_size)
    print('  Epochs:',dP.epochs)
    print('  Number of labels:', dP.numLabels)
    print('  Stop at Best Model based on validation:', dP.stopAtBest)
    print('  Save Best Model based on validation:', dP.saveBestModel)
    if dP.regressor:
        print('  Metric for Best Regression Model:', dP.metricBestModelR)
    else:
        print('  Metric for Best Classifier Model:', dP.metricBestModelC)
    #print('  ================================================\n')

#************************************
# Plot Activations in Predictions
#************************************
def plotActivationsTrain(model):
    pass

#************************************
# Plot Activations in Predictions
#************************************
def plotActivationsPredictions(R, model):
    pass

#************************************
# Lists the program usage
#************************************
def usage():
    print('\n Please make sure you have chosen at least one file.\n')
