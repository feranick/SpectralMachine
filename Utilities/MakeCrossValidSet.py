#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*************************************************
*
* Make Cross Validation Dataset from Learing Set
*
* version: 20170720a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
*************************************************
'''
print(__doc__)


import numpy as np
import sys, os.path, getopt, glob

class defParam:
    addToFlatland = False

def main():
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 makeCrossValidSet.py <learnData> <file-list-data>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    trainFile = os.path.splitext(sys.argv[1])[0] + '_train.txt'
    testFile = os.path.splitext(sys.argv[1])[0] + '_test.txt'

    if os.path.exists(trainFile) or os.path.exists(testFile) == True:
        print(" Training or Cross validation files exist. Exiting.\n")
        return

    En, M = readLearnFile(sys.argv[1])
    
    L = readList(sys.argv[2])

    newTrain = np.append([0], En)
    newTest = np.append([0], En)

    print(" Selecting:",str(L.shape[0])," spectra for test file (",str(L.shape[0]*100/M.shape[0]),"%)")

    print(" Sorting spectra for training and testing according to list...")
    for i in range(1,M.shape[0]):
        if i+1 in L:
            newTest = np.vstack((newTest, M[i]))
        else:
            newTrain = np.vstack((newTrain, M[i]))

    print('\n Saving new training file in', trainFile)
    with open(trainFile, 'ab') as f:
        np.savetxt(f, newTrain, delimiter='\t', fmt='%10.6f')

    print(' Saving new cross validation file in:', testFile, '\n')
    with open(testFile, 'ab') as f:
        np.savetxt(f, newTest, delimiter='\t', fmt='%10.6f')

    print(' Done!\n')

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

    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    #M = np.delete(M,np.s_[0:1],0)
    return En, M

#************************************
''' Open list '''
#************************************
def readList(File):
    try:
        with open(File, 'r') as f:
            L = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' List data file not found \n' + '\033[0m')
        return
    
    return L

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
