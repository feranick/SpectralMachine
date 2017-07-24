#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*************************************************
*
* Make Cross Validation Dataset from Learing Set
* Uses CSV with selected spectra from log file.
*
* version: 20170724d
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
*************************************************
'''
print(__doc__)


import numpy as np
import sys, os.path, getopt, glob

def main():
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 MakeCrossValidSet.py <learnData> <index_csv_data>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    trainFile = os.path.splitext(sys.argv[1])[0] + '_train.txt'
    testFile = os.path.splitext(sys.argv[1])[0] + '_test.txt'

    if os.path.exists(trainFile) or os.path.exists(testFile) == True:
        print(" Training or cross validation test files exist. Exiting.\n")
        return

    En, M = readLearnFile(sys.argv[1])
    I = readIndexFile(sys.argv[2])
    L = np.zeros(0)
    
    for i in range(0,I.shape[0]):
        if I[i] != 0:
            L = np.append(L,[i])

    cvSize = L.shape[0]*100/I.shape[0]
    
    print("\n Size of initial training set:", str(I.shape[0]),
          "\n Size of final training set:", str(I.shape[0]-L.shape[0]),
          "\n Size of final testing set: ",str(L.shape[0]),
          " ({:.2f}%)\n".format(cvSize))
        
    L = L.reshape(-1,1)
    newTrain = np.append([0], En)
    newTest = np.append([0], En)

    print(" Sorting spectra for training and testing according to list...")
    for i in range(0,M.shape[0]):
        if i in L:
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
    M = np.delete(M,np.s_[0:1],0)
    return En, M

#************************************
''' Open Index File '''
#************************************
def readIndexFile(File):
    try:
        csv = np.genfromtxt(File,delimiter=',')
        L = np.nan_to_num(csv[:,1])
    except:
        print('\033[1m' + ' Index data file not found \n' + '\033[0m')
        return
    L = np.delete(L,np.s_[0:1],0)
    return L

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
