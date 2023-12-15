#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************************
* InfoDatasets
* Info data for a given learning dataset
* version: v2023.12.15.1
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
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 InfoDatasets.py <learnData>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, M = readLearnFile(sys.argv[1])

    print(" Initial Energy: {0:.1f}, Final Energy: {1:.1f}, Step: {2:.1f}".format(En[0], En[-1], En[1]-En[0]))
    print(" Number of points per spectra:", M[0,1:].size)
    print("\n Number of unique classes:", np.unique(M[:,0]).size)
    print(" Number of spectra in training set:", M.shape[0],"\n")

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

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
