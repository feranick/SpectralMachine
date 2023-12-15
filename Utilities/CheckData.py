#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* Check whether a training file has data with
*  intensity higher than specified value
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)


import numpy as np
import sys, os.path, h5py, csv

def main():
    if(len(sys.argv)<3):
        print(' Usage:\n  python3 CheckData.py <learnData> <value>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    print(" Reading:",sys.argv[1],"\n")
    En, M = readLearnFile(sys.argv[1])

    for i in range(1, M.shape[0]):
        isMax = 0
        for l in M[i,1:]:
            if l > float(sys.argv[2]):
                isMax = 1
        if isMax == 1:
            print(" Row:\033[1m",i+2,"\033[0mexceeds",sys.argv[2],"in intensity")

    print("\n Done\n")

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
    return En, A

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
