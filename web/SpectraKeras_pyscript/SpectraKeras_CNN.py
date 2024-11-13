#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************
* SpectraKeras_CNN Classifier and Regressor
* Pyscript version
* Only for prediction with tflite_runtime
* v2024.11.13.1
* Uses: TFlite_runtime
* By: Nicola Ferralis <feranick@hotmail.com>
**********************************************
'''
#print(__doc__)

import numpy as np
import sys, os.path, configparser, js
from pyscript import fetch, document
import _pickle as pickle
#import pickle

from libSpectraKeras import *

baseUrl = "https://gridedgedm.mit.edu/SpectraKeras_pyscript/"

#************************************
# Parameters
#************************************
class Conf():
    def __init__(self, folder, ini):
    
        self.conf = configparser.ConfigParser()
        self.conf.optionxform = str
        
        self.readConfig(ini)
        self.model_directory = folder+"/"
        
        if self.regressor:
            self.modelName = "model_regressor_CNN.tflite"
        else:
            self.predProbThreshold = 90.00
            self.modelName = "model_classifier_CNN.tflite"

        self.tb_directory = "model_CNN"
        self.model_name = self.model_directory+self.modelName
        
        self.model_le = self.model_directory+"model_le.pkl"
        self.spectral_range = "model_spectral_range.pkl"
        
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
        
            self.SKDef = self.conf['Parameters']
            self.sysDef = self.conf['System']

            self.regressor = self.conf.getboolean('Parameters','regressor')
            self.normalize = self.conf.getboolean('Parameters','normalize')
            self.l_rate = self.conf.getfloat('Parameters','l_rate')
            self.l_rdecay = self.conf.getfloat('Parameters','l_rdecay')
            self.CL_filter = eval(self.SKDef['CL_filter'])
            self.CL_size = eval(self.SKDef['CL_size'])
            self.max_pooling = eval(self.SKDef['max_pooling'])
            self.dropCNN = eval(self.SKDef['dropCNN'])
            self.HL = eval(self.SKDef['HL'])
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

async def getFile(folder, file, bin):
    url = baseUrl+folder+"/"+file
    if bin:
        data = await fetch(url).bytearray()
    else:
        data = await fetch(url).text()
    return data
    
async def getModel(event):
    global df, folder
    document.querySelector("#button").disabled = True
    document.querySelector("#model").disabled = True
    
    if document.querySelector("#model").selectedIndex == 0:
        folder = "model-raman"
    else:
        folder = "model-raman"
    ini = await getFile(folder, "SpectraKeras_CNN.ini", False)
    dP = Conf(folder, ini)
    #modelPkl = await getFile(folder, dP.modelName, True)
    #df = pickle.loads(modelPkl)
    #output_div.innerHTML = ""
    document.querySelector("#button").disabled = False
    document.querySelector("#model").disabled = False

#************************************
# Main
#************************************
async def main(event):
    output_div = document.querySelector("#output")
    output_div.innerHTML = "Please wait..."

    ini = await getFile(folder, "SpectraKeras_CNN.ini", False)
    dP = Conf(folder, ini)
    Rtot = np.asarray(js.spectra.to_py())
    R = Rtot[:,1]
    Rx = Rtot[:,0]
    
    js.console.log(f'{Rtot}')
    js.console.log(f'{R}')
    js.console.log(f'{Rx}')
    
    output_div.innerHTML = R
    
    '''
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
            print('  Prediction\t| Probability [%]')
            print('  ----------------------------- ')
            for i in range(len(predictions[0])-1):
                if predictions[0][i]>0.01:
                    if dP.useTFlitePred:
                        print("  {0:d}\t\t| {1:.2f}".format(int(le.inverse_transform(i)[0]),100*predictions[0][i]/255))
                    else:
                        print("  {0:d}\t\t| {1:.2f}".format(int(le.inverse_transform(i)[0]),100*predictions[0][i]))
            print('\n  Predicted value = {0:d} (probability = {1:.2f}%)\n'.format(int(predValue), predProb))
            print('  ========================================================\n')

        else:
            print('\n ==========================================')
            print('\n Predicted value (probability = ' + str(predProb) + '%)')
            print(' ==========================================\n')
            print("  1:", str(predValue[0]),"%")
            print("  2:",str(predValue[1]),"%")
            print("  3:",str((predValue[1]/0.5)*(100-99.2-.3)),"%\n")
            print(' ==========================================\n')

    if dP.plotActivations and not dP.useTFlitePred:
        plotActivationsPredictions(R,model)
    '''

#************************************
# Accuracy determination
#************************************
def accDeterm(testFile):
    dP = Conf()
    model = loadModel(dP)
    En, A, Cl = readLearnFile(testFile, dP)
    predictions = np.zeros((0,0))
    fileName = []

    print("\n  Number of spectra in testing file:",A.shape[0])

    for row in A:
        R = formatForCNN(np.array([row]))
        try:
            predictions, _ = np.vstack((predictions,getPredictions(R, model, dP).flatten()))
        except:
            predictions, _ = np.array([getPredictions(R, model,dP).flatten()])

    if dP.regressor:
        print("\n  Accuracy determination is not defined in regression. Exiting.\n")
        return
    else:
        with open(dP.model_le, "rb") as f:
            le = pickle.load(f)
        summaryFile = np.array([['SpectraKeras_CNN','Classifier',''],['Real Class','Predicted Class', 'Probability']])

        successPred = 0

        for i in range(predictions.shape[0]):
            pred_class = np.argmax(predictions[i])
            if dP.useTFlitePred:
                predProb = round(100*predictions[i][pred_class]/255,2)
            else:
                predProb = round(100*predictions[i][pred_class],2)
            rosterPred = np.where(predictions[i][0]>0.1)[0]

            if pred_class.size >0:
                predValue = le.inverse_transform(pred_class)[0]
            else:
                predValue = 0
            if Cl[i] == predValue:
                successPred+=1

            summaryFile = np.vstack((summaryFile,[Cl[i], predValue, predProb]))

    print("\n\033[1m  Overall average accuracy: {0:.2f}% \033[0m".format(successPred*100/Cl.shape[0]))
    summaryFile[0,2] = "Av Acc: {0:.2f}%".format(successPred*100/Cl.shape[0])

    import pandas as pd
    df = pd.DataFrame(summaryFile)
    df.to_csv(dP.summaryAccFileName, index=False, header=False)
    print("\n  Prediction summary saved in:",dP.summaryAccFileName,"\n")


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
