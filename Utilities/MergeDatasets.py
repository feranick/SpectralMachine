#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* MergeDatasets
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
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
        print(' Usage:\n  python3 mergeDatasets.py <learnData1> <learnData2>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    newFile = os.path.splitext(sys.argv[1])[0] + '_merged_' + os.path.splitext(sys.argv[2])[0]
    learnFileExt = os.path.splitext(sys.argv[1])[1]

    En, M1 = readLearnFile(sys.argv[1])
    En2, M2 = readLearnFile(sys.argv[2])
    M = np.append([0], En)

    if En.shape[0] == En2.shape[0]:
        print(" Number of points in the learning datasets:", En.size)
        M = np.vstack((M,np.vstack((M1,M2))))
    else:
        print(" Number of points per spectra (1):", En.size)
        print(" Number of points per spectra (2):", En2.size)
        print(" Converting number of points per spectra (2) into that of spectra (1)...")
        M = np.vstack((M,M1))
        for i in range(M2.shape[0]):
            R = np.append(float(M2[i,0]), np.interp(En, En2, M2[i,1:], left = 0, right = 0))
            M = np.vstack((M,R))
        print(" Number of points per spectra (merged):", En.size)

    print("\n Original number of unique classes (1):", np.unique(M1[:,0]).size)
    print(" Original number of unique classes (2):", np.unique(M2[:,0]).size)
    print(" Number of unique classes (merged):", np.unique(M[:,0]).size)

    print("\n Original number of spectra in training set (1):", M1.shape[0])
    print(" Original number of spectra in training set (2):", M2.shape[0])
    print(" Original number of spectra in training set (merged):", M.shape[0]-1,"\n")

    saveLearnFile(M, newFile)

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
