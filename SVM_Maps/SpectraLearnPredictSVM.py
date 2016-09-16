#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
*********************************************
*
* SpectraLearnPredictSVM
* Perform SVM machine learning on Raman maps.
* version: 20160916e
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

#**********************************************
''' Input/Output files '''
#**********************************************
trainedData = "trained.pkl"

#**********************************************
''' Training algorithm
    Use either 'linear' or 'rbf'
    ('rbf' for large number of features) '''
#**********************************************
Cfactor = 5
kernel = 'rbf'

#**********************************************
''' Spectra normalization conditions '''
#**********************************************
Ynorm = True
YnormTo = 10
YnormX = 534

#**********************************************
''' Plotting options '''
#**********************************************
showProbPlot = True
showTrainingDataPlot = False

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
    f = open(mapfile, 'r')
    M = np.loadtxt(f, unpack =False)
    f.close()
        
    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(M,np.s_[0:1],0)
    Cl = ['{:.2f}'.format(x) for x in M[:,0]]
    A = np.delete(M,np.s_[0:1],1)


    if Ynorm == True:
        print(' Normalizing spectral intensity to: ' + str(YnormTo) + '; En(' + str(YnormX) + ') = ' + str(En[YnormX]) + '\n')
        for i in range(0,A.shape[0]):
            A[i,:] = np.multiply(A[i,:], YnormTo/A[i,YnormX])


    print(' Number of datapoints = ' + str(A.shape[0]))
    print(' Size of each datapoints = ' + str(A.shape[1]) + '\n')

    #**********************************************
    ''' Load trained files or retrain '''
    #**********************************************
    try:
        with open(trainedData):
            print(" Opening training data...")
            clf = joblib.load(trainedData)
    except:
        print(' Retraining data...')
        clf = svm.SVC(kernel = kernel, C = Cfactor, decision_function_shape = 'ovr', probability=True)
        clf.fit(A,Cl)
        Z= clf.decision_function(A)
        print(' Number of classes = ' + str(Z.shape[1]))
        joblib.dump(clf, trainedData)

    #**********************************************
    ''' Run prediction '''
    #**********************************************
    f = open(sampleFile, 'r')
    R = np.loadtxt(f, unpack =True, usecols=range(1,2))
    R = R.reshape(1,-1)
    f.close()

    if Ynorm == True:
            R[0,:] = np.multiply(R[0,:], YnormTo/R[0,YnormX])

    print('\n Predicted value = ' + str(clf.predict(R)[0]))
    prob = clf.predict_proba(R)[0].tolist()

    #print(' Probabilities of this sample within each class: \n')
    #for i in range(0,clf.classes_.shape[0]):
    #   print(' ' + str(clf.classes_[i]) + ': ' + str(round(100*prob[i],2)) + '%')

    #********************
    ''' Plot results '''
    #********************
    if showProbPlot == True:
        print('\n Stand by: Plotting probabilities for each class...')
        plt.title('Probability density per class: ' + str(sampleFile))
        for i in range(0, clf.classes_.shape[0]):
            plt.scatter(clf.classes_[i], round(100*prob[i],2), label='probability', c = 'red')
        plt.grid(True)
        plt.xlabel('Class')
        plt.ylabel('Probability [%]')
        plt.show()

    if showTrainingDataPlot == True:
        print(' Stand by: Plotting each datapoint from the map...')
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
