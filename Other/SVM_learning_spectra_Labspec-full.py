#!/usr/bin/python
# -*- coding: utf-8 -*-

#*********************************************
#
# SVM_learning_spectra_Labspec-full
# Perform SVM machine learning on Raman maps.
# version: 20160926a
#
# By: Nicola Ferralis <feranick@hotmail.com>
#
#**********************************************

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm
from sklearn.externals import joblib


sampleFile = "Sample.txt"
whichColumn = 2

mapfile = "map.txt"
clustfile = "clust.txt"
trainedData = "trained.pkl"
kernel = 'rbf'  #Use either 'linear' or 'rbf' (for large number of features)

#clf = svm.SVC(kernel = 'linear', C = 1.0)
#clf.fit(A,Cl)

try:
    with open(trainedData):
        print(" Opening training data...")
        clf = joblib.load(trainedData)
except:
        print(" Opening learning files and parameters...")
        f = open(mapfile, 'r')
        A = np.loadtxt(f, unpack =False, skiprows=1)
        A = np.delete(A, np.s_[0:2], 1)
        f.close()
        print(A)
        print(A.shape)

        f = open(clustfile, 'r')
        Cl = np.loadtxt(f, unpack =False, skiprows=1, usecols = range(whichColumn-1,whichColumn))
        f.close()
        #Cl = [round(x*1000) for x in Cl]
        Cl = ['{:.2f}'.format(x) for x in Cl]
        print(Cl)

        print(' Retraining data...\n')
        clf = svm.SVC(kernel = kernel, C = 1.0)
        clf.fit(A,Cl)
        joblib.dump(clf, trainedData)

f = open(sampleFile, 'r')
R = np.loadtxt(f, unpack =True, usecols=range(1,2))
R = R.reshape(1,-1)
f.close()

print('\n Predicted value = ' + str(clf.predict(R)[0]) + '\n')
