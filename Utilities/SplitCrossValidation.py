#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Split learning data into Train/Test
* version: 20171219a
*
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)
import numpy as np
import sys, os.path, random
import matplotlib.pyplot as plt

def main():
    if(len(sys.argv)<2):
        print(' Usage:\n  python3 SplitCrossValidation.py <learnData> <percentage>\n')
        print(' Usage (2.6%):\n  python3 PlotData.py <learnData>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    try:
        perc = float(sys.argv[2])
    except:
        perc = 0.026

    En, M, Cl, learnFileRoot = readLearnFile(sys.argv[1])
    print("Splitting learning file into train/test for cross-validation: ",str(perc*100),"%\n")
    M_train, Cl_train, M_test, Cl_test =  formatSubset(M, Cl, perc)
    saveSubsets(En, M_train, Cl_train, learnFileRoot+"_train.txt", "training")
    saveSubsets(En, M_test, Cl_test, learnFileRoot+"_test.txt", "testing")

#************************************
''' Open Learning Data '''
#************************************
def readLearnFile(learnFile):
    try:
        with open(learnFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Learn data file not found \n' + '\033[0m')
        return

    learnFileRoot = os.path.splitext(learnFile)[0]

    #En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    #M = np.delete(np.array(M[:,1:]),np.s_[0:1],0)
    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(M,np.s_[0:1],0)
    Cl = np.asarray(['{:.2f}'.format(x) for x in M[:,0]]).reshape(-1,1)
    M = np.delete(M,np.s_[0:1],1)
    
    print("En:",En.shape)
    print("M:",M.shape)
    return En, M, Cl, learnFileRoot

####################################################################
''' Format subset of training data '''
####################################################################
def formatSubset(A, Cl, percent):
    from sklearn.model_selection import train_test_split
    A_train, A_cv, Cl_train, Cl_cv = \
    train_test_split(A, Cl, test_size=percent, random_state=42)
    return A_train, Cl_train, A_cv, Cl_cv

#************************************
''' Open Learning Data '''
#************************************
def saveSubsets(En, M, Cl, file, type):
    M = np.hstack((M, Cl))
    if os.path.exists(file) == False:
        newFile = np.append([0], En)
        newFile = np.vstack((newFile, M))
    else:
        newFile = M
    with open(file, 'ab') as f:
        np.savetxt(f, newFile.astype(float), delimiter='\t', fmt='%10.6f')

    print(' New',type,'file saved:', file, '\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
