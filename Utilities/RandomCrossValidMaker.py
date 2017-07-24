#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* Create Random Cross Validation Datasets
* Train + Test
*
* version: 20170724a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)


import numpy as np
import sys, os, getopt, glob, csv

def main():
    if(len(sys.argv)<3):
        print(' Usage:\n  python3 RandomCrossValidMaker.py <learnData> <percentageCrossValid>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    print(' Opening initial training file:', sys.argv[1])
    En, A, Cl = readLearnFile(sys.argv[1])

    percTrain = str('{:.0f}'.format(100-float(sys.argv[2])))
    percTest = str('{:.0f}'.format(float(sys.argv[2])))
                    
    newTrainFile = os.path.splitext(sys.argv[1])[0] + '_train-' + percTrain + 'pc.txt'
    newTestFile = os.path.splitext(sys.argv[1])[0] + '_test-' + percTest + 'pc.txt'

    print(' Splitting', sys.argv[1], ' (Train:', percTrain,'%; Test:',percTest,'%)\n')
    A_train, Cl_train, A_test, Cl_test = formatSubset(A, Cl, float(sys.argv[2])/100)

    print(' Writing new training file: ', newTrainFile)
    writeFile(newTrainFile, En, A_train, Cl_train)
    print('\n Writing new cross-validation file: ', newTestFile)
    writeFile(newTestFile, En, A_test, Cl_test)

    print('\n Done!\n')

#************************************
''' Read Learning file '''
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
    Cl = ['{:.2f}'.format(x) for x in M[:,0]]
    A = np.delete(M,np.s_[0:1],1)

    print(' Number of datapoints = ' + str(A.shape[0]))
    print(' Size of each datapoint = ' + str(A.shape[1]) + '\n')
    return En, A, Cl

#*****************************************
''' Write new learning and test files '''
#*****************************************
def writeFile(File, En, A, Cl):
    print(' Number of datapoints:', str(A.shape[0]))
    with open(File, 'ab') as f:
        if os.stat(File).st_size == 0:
            np.savetxt(f, np.append([0], En).reshape(1,-1), fmt='%5s')
        for i in range(len(Cl)):
            np.savetxt(f, np.append(Cl[i], A[i]).reshape(1,-1), fmt='%5s')

#************************************
''' Format subset '''
#************************************
def formatSubset(A, Cl, percent):
    from sklearn.model_selection import train_test_split
    A_train, A_cv, Cl_train, Cl_cv = \
    train_test_split(A, Cl, test_size=percent, random_state=42)
    return A_train, Cl_train, A_cv, Cl_cv

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
