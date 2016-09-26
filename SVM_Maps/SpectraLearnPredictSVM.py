#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
*********************************************
*
* SpectraLearnPredictSVM
* Perform SVM machine learning on Raman maps.
* version: 20160926a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
**********************************************
'''
print(__doc__)

import matplotlib
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Agg')

import numpy as np
from sklearn import svm
import sys, os.path
from sklearn.externals import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, RandomizedPCA

#**********************************************
''' Input/Output files '''
#**********************************************
trainedData = "trained.pkl"
alwaysRetrain = True

#**********************************************
''' Training algorithm
    Use either 'linear' or 'rbf'
    ('rbf' for large number of features) '''
#**********************************************
Cfactor = 10
kernel = 'rbf'
showClasses = False

#**********************************************
''' Spectra normalization conditions '''
#**********************************************
Ynorm = True
YnormTo = 1
YnormX = 1604
YnormXdelta = 30
# Normalize from the full spectra (False: recommended)
fullYnorm = False 

#**********************************************
''' Plotting options '''
#**********************************************
showProbPlot = False
showTrainingDataPlot = False

#**********************************************
''' Principal component analysis (PCA) '''
#**********************************************
runPCA = False
numPCAcomp = 5

#**********************************************
''' Main '''
#**********************************************
def main():
    try:
        LearnPredict(sys.argv[1], sys.argv[2])
    except:
        usage()
        sys.exit(2)

#**********************************************
''' Learn and Predict '''
#**********************************************
def LearnPredict(mapFile, sampleFile):
    
    #**********************************************
    ''' Open and process training data '''
    #**********************************************
    try:
        with open(mapFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Map data file not found \n' + '\033[0m')
        return

    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(M,np.s_[0:1],0)
    Cl = ['{:.2f}'.format(x) for x in M[:,0]]
    A = np.delete(M,np.s_[0:1],1)

    # Find index corresponding to energy value to be used for Y normalization
    if fullYnorm == False:
        YnormXind = np.where((En<float(YnormX+YnormXdelta)) & (En>float(YnormX-YnormXdelta)))[0].tolist()
    else:
        YnormXind = np.where(En>0)[0].tolist()

    Amax = np.empty([A.shape[0],1])
    print(' Number of datapoints = ' + str(A.shape[0]))
    print(' Size of each datapoint = ' + str(A.shape[1]) + '\n')

    #**********************************************
    ''' Normalize if flag is set '''
    #**********************************************
    if Ynorm == True:
        print(' Normalizing spectral intensity to: ' + str(YnormTo) + '; En = [' + str(YnormX-YnormXdelta) + ', ' + str(YnormX+YnormXdelta) + ']\n')
        for i in range(0,A.shape[0]):
            Amax[i] = A[i,A[i][YnormXind].tolist().index(max(A[i][YnormXind].tolist()))+YnormXind[0]]
            A[i,:] = np.multiply(A[i,:], YnormTo/Amax[i])
    
    try:
        if alwaysRetrain == False:
            with open(trainedData):
                print(" Opening training data...")
                clf = joblib.load(trainedData)
        else:
            raise ValueError('Force retraining.')
    except:
        
        #**********************************************
        ''' Retrain data if not available'''
        #**********************************************
        print(' Retraining data...')
        clf = svm.SVC(kernel = kernel, C = Cfactor, decision_function_shape = 'ovr', probability=True)
        clf.fit(A,Cl)
        Z= clf.decision_function(A)
        print(' Number of classes = ' + str(Z.shape[1]))
        joblib.dump(clf, trainedData)
        if showClasses == True:
            print(' List of classes: ' + str(clf.classes_))

    #**********************************************
    ''' Run prediction '''
    #**********************************************
    try:
        with open(sampleFile, 'r') as f:
            print(' Opening sample data for prediction...')
            R = np.loadtxt(f, unpack =True, usecols=range(1,2))
    except:
        print('\033[1m' + '\n Sample data file not found \n ' + '\033[0m')
        return

    R = R.reshape(1,-1)

    if(R.shape[1] != A.shape[1]):
        print('\033[1m' + '\n WARNING: Prediction aborted. Different number of datapoints\n for the energy axis for training (' + str(A.shape[1]) + ') and sample (' + str(R.shape[1]) + ') data.\n Reformat sample data with ' + str(A.shape[1]) + ' X-datapoints.\n' + '\033[0m')
        return

    if Ynorm == True:
        Rmax = R[0,R[0][YnormXind].tolist().index(max(R[0][YnormXind].tolist()))+YnormXind[0]]
        R[0,:] = np.multiply(R[0,:], YnormTo/Rmax)

    print('\n Predicted value = ' + str(clf.predict(R)[0]) +'\n')
    prob = clf.predict_proba(R)[0].tolist()

    #print(' Probabilities of this sample within each class: \n')
    #for i in range(0,clf.classes_.shape[0]):
    #   print(' ' + str(clf.classes_[i]) + ': ' + str(round(100*prob[i],2)) + '%')


    #********************
    ''' Run PCA '''
    #********************
    if runPCA == True:
        print(' Running PCA...\n')
        pca = PCA(n_components=numPCAcomp)
        pca.fit_transform(A)
        for i in range(0,pca.components_.shape[0]):
            plt.plot(En, pca.components_[i,:], label=i)
        #plt.plot(En, pca.components_[1,:]-pca.components_[0,:], label='Difference')
        plt.xlabel('Raman shift [1/cm]')
        plt.ylabel('PCA')
        plt.legend()
        plt.show()


    #********************
    ''' Plot results '''
    #********************
    if showProbPlot == True:
        print('\n Stand by: Plotting probabilities for each class... \n')
        plt.title('Probability density per class: ' + str(sampleFile))
        for i in range(0, clf.classes_.shape[0]):
            plt.scatter(clf.classes_[i], round(100*prob[i],2), label='probability', c = 'red')
        plt.grid(True)
        plt.xlabel('Class')
        plt.ylabel('Probability [%]')
        plt.show()

    if showTrainingDataPlot == True:
        print(' Stand by: Plotting each datapoint from the map...\n')
        if Ynorm ==True:
            plt.title("Normalized Training Data")
        else:
            plt.title("Training Data")
        for i in range(0,A.shape[0]):
            plt.plot(En, A[i,:], label='Training data')
            plt.plot(En, R[0,:], linewidth = 2, label='Sample data')
            plt.xlabel('Raman shift [1/cm]')
            plt.ylabel('Raman Intensity [arb. units]')
        plt.show()

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:')
    print('  python SpectraLearnPredictSVM.py <mapfile> <spectrafile> \n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
