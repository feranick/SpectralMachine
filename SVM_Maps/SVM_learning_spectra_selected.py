#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

#*********************************************
#
# SVM_learning_spectra_selected
# Perform SVM machine learning on Raman maps.
# version: 20160914a
#
# By: Nicola Ferralis <feranick@hotmail.com>
#
#**********************************************

import numpy as np
from sklearn import svm
from sklearn.externals import joblib


sampleFile = "Sample.txt"

mapfile = "Dracken-7-tracky_map1_bs_fit2_selected.txt"
trainedData = "trained.pkl"

try:
    with open(trainedData):
        print(" Opening training data...")
        clf = joblib.load(trainedData)
except:
        print(" Opening learning files and parameters...")
        f = open(mapfile, 'r')
        M = np.loadtxt(f, unpack =False)
        #A = np.delete(A, np.s_[0:2], 1)
        f.close()
        #print(M)
        #print(M.shape)

        Cl = [round(x*100) for x in M[:,0]]
        A = np.delete(M,np.s_[0:1],1)
        print(A)
        print(Cl)
        print(' Retraining data...\n')
        clf = svm.SVC(kernel = 'linear', C = 1.0)
        clf.fit(A,Cl)
        joblib.dump(clf, trainedData)

f = open(sampleFile, 'r')
R = np.loadtxt(f, unpack =True, usecols=range(1,2))
R = R.reshape(1,-1)
f.close()

print(" Prediction in progress...")
print(clf.predict(R))