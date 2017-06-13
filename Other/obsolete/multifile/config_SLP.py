#!/usr/bin/env python
# -*- coding: utf-8 -*-



#**********************************************
''' Calculation by limited number of points '''
#**********************************************
cherryPickEnPoint = False  # False recommended

enSel = [1050, 1150, 1220, 1270, 1330, 1410, 1480, 1590, 1620, 1650]
enSelDelta = [2, 2, 2, 2, 10, 2, 2, 15, 5, 2]

if(cherryPickEnPoint == True):
    enRestrictRegion = False
    print(' Calculation by limited number of points: ENABLED ')
    print(' THIS IS AN EXPERIMENTAL FEATURE \n')
    print(' Restricted range: DISABLED')

#**********************************************
''' Spectra normalization, preprocessing '''
#**********************************************
Ynorm = True   # True recommended
YnormTo = 1
YnormX = 1600
YnormXdelta = 30

fullYnorm = False  # Normalize full spectra (False: recommended)

preProcess = True  # True recommended

enRestrictRegion = False
enLim1 = 450    # for now use indexes rather than actual Energy
enLim2 = 550    # for now use indexes rather than actual Energy

#**********************************************
''' Model selection for training '''
#**********************************************
modelSelection = False
percentCrossValid = 0.05

#**********************************************
''' Support Vector Classification'''
#**********************************************
runSVM = True
svmClassReport = False

svmTrainedData = "svmModel.pkl"
class svmDef:
    svmAlwaysRetrain = True
    plotSVM = True

''' Training algorithm for SVM
Use either 'linear' or 'rbf'
('rbf' for large number of features) '''

Cfactor = 20
kernel = 'rbf'
showClasses = False

#**********************************************
''' Neural Networks'''
#**********************************************
runNN = True
nnClassReport = False

nnTrainedData = "nnModel.pkl"
class nnDef:
    nnAlwaysRetrain = True
    plotNN = True

''' Solver for NN
    lbfgs preferred for small datasets
    (alternatives: 'adam' or 'sgd') '''
nnSolver = 'lbfgs'
nnNeurons = 100  #default = 100

#**********************************************
''' TensorFlow '''
#**********************************************
runTF = True

tfTrainedData = "tfmodel.ckpt"
class tfDef:
    tfAlwaysRetrain = False
    plotTF = True

#**********************************************
''' Principal component analysis (PCA) '''
#**********************************************
runPCA = False
customNumPCAComp = True
numPCAcomponents = 2

#**********************************************
''' K-means '''
#**********************************************
runKM = False
customNumKMComp = False
numKMcomponents = 20

class kmDef:
    plotKM = True
    plotKMmaps = True

#**********************************************
''' Plotting '''
#**********************************************
showProbPlot = False
showTrainingDataPlot = False
showPCAPlots = True

#**********************************************
''' Multiprocessing '''
#**********************************************
multiproc = False
