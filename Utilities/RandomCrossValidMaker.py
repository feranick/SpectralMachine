#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* Create Random Cross Validation Datasets
* Train + Test
* version: v2024.10.07.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os, csv, h5py

class defParam:
    saveAsTxt = False
    randomSel = None #None or any seed, like 42.

def main():
    if(len(sys.argv)<3):
        print(' Usage:\n  python3 RandomCrossValidMaker.py <learnData> <percentageCrossValid>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, A, Cl = readLearnFile(sys.argv[1])

    percTrain1 = str('{:.1f}'.format(100-float(sys.argv[2])))
    percTest1 = str('{:.1f}'.format(float(sys.argv[2])))

    percTrain = str('{:1d}-{:1d}'.format(int(100-float(sys.argv[2])),int((100-float(sys.argv[2])-int(100-float(sys.argv[2])))*10)))
    percTest = str('{:1d}-{:1d}'.format(int(float(sys.argv[2])),int((float(sys.argv[2])-int(float(sys.argv[2])))*10)))
    
    newTrainFile = os.path.splitext(sys.argv[1])[0] + '_train-cv' + percTrain + 'pc'
    newTestFile = os.path.splitext(sys.argv[1])[0] + '_test-cv' + percTest + 'pc'

    if defParam.saveAsTxt == True:
        newTrainFile = newTrainFile + '.txt'
        newTestFile = newTestFile + '.txt'
    else:
        newTrainFile = newTrainFile + '.h5'
        newTestFile = newTestFile + '.h5'

    if os.path.exists(newTrainFile) or os.path.exists(newTestFile) == True:
        print(" Training or cross validation test files exist. Exiting.\n")
        return

    print(' Splitting', sys.argv[1], ' (Train:', percTrain1,'%; Test:',percTest1,'%)\n')
    A_train, Cl_train, A_test, Cl_test, flag = formatSubset(A, Cl, float(sys.argv[2])/100)

    if flag:
        print("\n Would you like to save dataset with multipe data per class (0: Yes, 1: No)")
        us = input()
        if us == "1":
            return
    
    print('\n Writing new training file: ', newTrainFile)
    writeFile(newTrainFile, En, A_train, Cl_train)
    print('\n Writing new cross-validation file: ', newTestFile)
    writeFile(newTestFile, En, A_test, Cl_test)
    print('\n Done!\n')
    
#************************************
''' Format subset '''
#************************************
def formatSubset(A, Cl, percent):
    from sklearn.model_selection import train_test_split
    A_train, A_cv, Cl_train, Cl_cv = \
    train_test_split(A, Cl, test_size=percent, random_state=defParam.randomSel)
    uniCl = np.unique(Cl_cv).astype(int)
    if Cl_cv.shape[0] - uniCl.shape[0] > 0:
        print(" Classes with multiple data present.")
        print("\n Unique classes in learning/validation set and corresponding number of members:\n")
        uni = np.ones(np.unique(Cl_cv).shape)
        for x in enumerate(uniCl):
            uni[x[0]] = np.count_nonzero(Cl_cv==uniCl[x[0]])
            if uni[x[0]] == 1:
                print(" {0:.0f}: {1:.0f} ".format(x[0],uni[x[0]]))
            else:
                print(" \033[1m {0:.0f}: {1:.0f} \033[0m".format(x[0],uni[x[0]]))
        flag = True
    else:
        print("\n Unique classes in learning/validation set:\n")
        with np.printoptions(threshold=np.inf):
            print(np.unique(Cl_cv).astype(int))
        flag = False
    return A_train, Cl_train, A_cv, Cl_cv, flag

def formatSubset2(A, Cl, percent):
    list = np.random.choice(range(len(Cl)), int(np.rint(percent*len(Cl))), replace=False)
    A_train = np.delete(A,list,0)
    Cl_train = np.delete(Cl,list)
    A_cv = A[list]
    Cl_cv = Cl[list]
    return A_train, Cl_train, A_cv, Cl_cv

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
            hf.create_dataset("M",  data=newMatrix.astype(np.float64))

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
