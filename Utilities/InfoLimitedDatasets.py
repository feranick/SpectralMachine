#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************************
* InfoLimitedDatasets
* Info data with little representation based on threshold
* version: 20181219a
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py
#************************************
''' Main '''
#************************************
def main():
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 InfoLimitedDatasets.py <learnData> <threshold>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, M = readLearnFile(sys.argv[1])

    numClasses = np.unique(M[:,0]).size
    indClass = np.zeros((numClasses))
    totNumIncl = 0

    # sort how many spectra per class
    for i in range(M.shape[0]):
        indClass[int(M[i,0])]+=1

    # identify the number of spectra above threshold
    for i in range(M.shape[0]):
        if indClass[int(M[i,0])] >= float(sys.argv[2]):
            totNumIncl += 1

    # find out how many classes are above the threshold
    for i in range(numClasses):
        if indClass[i] >= float(sys.argv[2]):
            print(" Class: ",i, "- spectra per class:", int(indClass[i]))
        else:
            print(" Class: ",i, "- spectra per class:", int(indClass[i])," - EXCLUDED")

    totClassIncl = indClass[np.where(indClass < float(sys.argv[2]))]

    print("\n Number of points per spectra:", M[0,1:].size)
    print("\n Original number of unique classes:", numClasses)
    print(" Number of included unique classes:",totClassIncl.shape[0])
    print(" Number of excluded unique classes:",indClass.shape[0] - totClassIncl.shape[0])

    print("\n Original number of spectra in training set:", M.shape[0])
    print(" Number of spectra included in new training set:", totNumIncl)
    print(" Number of spectra excluded in new training set:", M.shape[0] - totNumIncl,"\n")


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
