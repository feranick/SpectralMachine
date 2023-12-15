#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*************************************************
* Make Cross Validation Dataset from Learing Set
* Uses CSV with selected spectra from log file.
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
*************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py
#************************************
''' Main '''
#************************************
class defParam:
    saveAsTxt = True

def main():
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 MakeCrossValidSet.py <learnData> <index_csv_data>\n')
        print('  To select validation spectra, add a \"1\" in the second column')
        print('  for the corresponding spectra in the csv index file\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, M = readLearnFile(sys.argv[1])
    I = readIndexFile(sys.argv[2])
    L = np.zeros(0)

    for i in range(0,I.shape[0]):
        if I[i] != 0:
            L = np.append(L,[i])

    if len(L) ==0:
        print(" No test validation spectra have been specified in the csv file. Exiting.\n")
        return

    cvSize = L.shape[0]*100/I.shape[0]
    
    print("\n Size of initial training set:", str(I.shape[0]),
          "\n Size of final training set:", str(I.shape[0]-L.shape[0]),
          "\n Size of final testing set: ",str(L.shape[0]),
          " ({:.2f}%)\n".format(cvSize))

    if defParam.saveAsTxt == True:
        trainFile = os.path.splitext(sys.argv[1])[0] + "_train-cv{:.2f}".format(cvSize) + "pc.txt"
        testFile = os.path.splitext(sys.argv[1])[0] + "_test-cv{:.2f}".format(cvSize) + "pc.txt"
    else:
        trainFile = os.path.splitext(sys.argv[1])[0] + "_train-cv{:.2f}".format(cvSize) + "pc.h5"
        testFile = os.path.splitext(sys.argv[1])[0] + "_test-cv{:.2f}".format(cvSize) + "pc.h5"

    if os.path.exists(trainFile) or os.path.exists(testFile) == True:
        print(" Training or cross validation test files exist. Exiting.\n")
        return
        
    L = L.reshape(-1,1)
    newTest = np.append([0], En)
    
    print(" Sorting spectra for training and testing according to list...")
    for i in L:
        newTest = np.vstack((newTest, M[int(i)]))
    newTrain = M
    for i in L:
        newTrain = np.delete(newTrain, int(i), 0)

    saveCVFiles(newTrain, newTest, trainFile, testFile)

    print(' Done!\n')

#************************************
''' Open Learning Data '''
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

    return En, M, Cl

#***************************************
''' Save split CV Learning/test Data '''
#***************************************
def saveCVFiles(newTrain, newTest, trainFile, testFile):
    if defParam.saveAsTxt == True:
        print("\n Saving new training file (txt) in:", trainFile)
        with open(trainFile, 'ab') as f:
            np.savetxt(f, newTrain, delimiter='\t', fmt='%10.6f')
        print(" Saving new cross validation (txt) in:", testFile, '\n')
        with open(testFile, 'ab') as f:
            np.savetxt(f, newTest, delimiter='\t', fmt='%10.6f')
    else:
        print("\n Saving new training file (hdf5) in: "+trainFile)
        with h5py.File(trainFile, 'w') as hf:
            hf.create_dataset("M",  data=newTrain)
        print(" Saving new cross validation file (hfd5) in:"+testFile+"\n")
        with h5py.File(testFile, 'w') as hf:
            hf.create_dataset("M",  data=newTest)

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
