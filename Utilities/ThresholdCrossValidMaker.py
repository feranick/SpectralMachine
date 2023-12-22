#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************************
* Create Random Cross Validation Datasets from threshold
* Based on high frequency datapoints
* version: v2023.12.22.1
* By: Nicola Ferralis <feranick@hotmail.com>
*********************************************************
'''
print(__doc__)

import numpy as np
import sys, os, csv, h5py

class defParam:
    saveAsTxt = False
    randomSel = None #None or any seed, like 42.

def main():
    if(len(sys.argv)<4):
        print(' Usage:\n  python3 ThresholdCrossValidMaker.py <learnData> <HighFreq threshold> <tot num valid data>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, A, Cl = readLearnFile(sys.argv[1])

    A_train, Cl_train, A_test, Cl_test, totNumPoints = selectHFdata(A, Cl, int(sys.argv[2]), int(sys.argv[3]))
    
    newTrainFile = os.path.splitext(sys.argv[1])[0] + '_train-cv_hfsel' + sys.argv[2] + '_val' + str(totNumPoints)
    newTestFile = os.path.splitext(sys.argv[1])[0] + '_test-cv_hfsel' + sys.argv[2] + '_val' + str(totNumPoints)

    if defParam.saveAsTxt == True:
        newTrainFile = newTrainFile + '.txt'
        newTestFile = newTestFile + '.txt'
    else:
        newTrainFile = newTrainFile + '.h5'
        newTestFile = newTestFile + '.h5'
        
    if os.path.exists(newTrainFile) or os.path.exists(newTestFile) == True:
        print(" Training or cross validation test files exist. Exiting.\n")
        return
    
    print(' Splitting', sys.argv[1], 'into training/validation datasets\n')
    print(' Writing new training file:', newTrainFile)
    writeFile(newTrainFile, En, A_train, Cl_train)
    print('\n Writing new cross-validation file:', newTestFile)
    writeFile(newTestFile, En, A_test, Cl_test)
    print('\n Done!\n')
    
#************************************
''' Select high frequency datasets '''
#************************************
def selectHFdata(A, Cl, HFthreshold, totNumPoints):
    A_train = A
    Cl_train = Cl
    uHFCl = np.array([])
    uniCl = np.unique(Cl).astype(int)
    for x in enumerate(uniCl):
        if np.count_nonzero(Cl==uniCl[x[0]]) >= HFthreshold:
            uHFCl = np.append(uHFCl, uniCl[x[0]])
    print(" Classes with members with more than {0:.0f} elements: {1:.0f}".format(HFthreshold,uHFCl.shape[0]))
    print(" Number of classes to be selected for validation: ", totNumPoints)
    
    if uHFCl.shape[0] < totNumPoints:
        print("\n \033[1mNot enough classes to select {0:.0f} classes for validation\033[0m.\n  Setting max number of classes to {0:.0f}.".format(uHFCl.shape[0]))
        totNumPoints = uHFCl.shape[0]
        
    list = np.array([]).astype(int)
    listSelHF = np.random.choice(uHFCl, int(totNumPoints), replace=False)
    for i in listSelHF:
        list = np.append(list,np.random.choice(np.where(np.in1d(Cl,i) == True)[0],replace=False)).tolist()

    A_train = np.delete(A,list,0)
    Cl_train = np.delete(Cl,list)
    A_cv = A[list]
    Cl_cv = Cl[list]
    
    print("\n Generated validation set with {0:.0f} members, each from a class with at least {1:.0f} elements.".format(listSelHF.shape[0],uHFCl.shape[0]))

    return A_train, Cl_train, A_cv, Cl_cv, totNumPoints

#************************************
''' Read Learning file '''
#************************************
def readLearnFile(learnFile):
    print(" Opening learning file: "+learnFile+"\n")
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        elif os.path.splitext(learnFile)[1] == ".h5":
            with h5py.File(learnFile, 'r') as hf:
                M = hf["M"][:]
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
    except:
        print("\033[1m" + " Learning file not found \n" + "\033[0m")
        return
    En = M[0,1:]
    A = M[1:,1:]
    Cl = M[1:,0]
    return En, A, Cl

#*****************************************
''' Write new learning and test files '''
#*****************************************
def writeFile(File, En, A, Cl):
    print(' Number of datapoints:', str(A.shape[0]))
    newMatrix = np.append([0], En).reshape(1,-1)
    temp = np.hstack((Cl[np.newaxis].T, A))
    newMatrix = np.vstack((newMatrix, temp))

    if defParam.saveAsTxt == True:
        with open(File, 'ab') as f:
            np.savetxt(f, newMatrix, delimiter='\t', fmt='%10.6f')
    else:
        with h5py.File(File, 'w') as hf:
            hf.create_dataset("M",  data=newMatrix)

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
