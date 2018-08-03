#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* RemoveLimitedDatasets
* Remove data with little representation based on threshold
*
* version: 20180803c
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py
#************************************
''' Main '''
#************************************
class defParam:
    saveAsTxt = False

def main():
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 RemoveLimitedDatasets.py <learnData> <threshold>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    newFile = os.path.splitext(sys.argv[1])[0] + '_above' + sys.argv[2]
    exclFile = os.path.splitext(sys.argv[1])[0] + '_excludeBelow' + sys.argv[2]
    learnFileExt = os.path.splitext(sys.argv[1])[1]

    En, M = readLearnFile(sys.argv[1])
    newTrain = np.append([0], En)
    exclTrain = np.append([0], En)

    num = 0
    ind = 0
    tempTrain = M[0,:]
    exclIndex = []

    for i in range(M.shape[0]):
        #print("initial: ", M[i,0], num)
        if M[i,0] != ind or i == M.shape[0]-1:
            if i == M.shape[0]-1:
                num += 1
            if num >= float(sys.argv[2]):
                print(" Class: ",ind, "\t- number per class:", num)
                newTrain = np.vstack((newTrain,tempTrain))
            else:
                print(" Class: ",ind, "\t- number per class:", num, " EXCLUDED")
                exclIndex.append(ind)
                exclTrain = np.vstack((exclTrain,tempTrain))
            tempTrain = M[i,:]
            ind = M[i,0]
            num = 1
        else:
            tempTrain=np.vstack((tempTrain,M[i,:]))
            num +=1

    print("\n Original number of spectra in training set:", M.shape[0])
    print(" Number of spectra included in new training set:", newTrain.shape[0]-1)
    print(" Number of spectra excluded in new training set:", exclTrain.shape[0]-1,"\n")

    saveLearnFile(newTrain, newFile)
    saveLearnFile(exclTrain, exclFile)

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
    A = M[1:,:]
    #Cl = M[1:,0]
    return En, A

#***************************************
''' Save new learning Data '''
#***************************************
def saveLearnFile(M, learnFile):
    if defParam.saveAsTxt == True:
        learnFile += '.txt'
        print(" Saving new training file (txt) in:", learnFile+"\n")
        with open(learnFile, 'ab') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    else:
        learnFile += '.h5'
        print(" Saving new training file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M)

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
