#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Load train data from binary file
* version: 20180203a
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
        print(' Usage:\n  python3 LoadBinary.py <learnData> <step>\n')
        print(' Usage (full set):\n  python3 LoadBinary.py <learnData>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    try:
        step = int(sys.argv[2])
    except:
        step = 1
    En, M, learnFileRoot = readLearnFile(sys.argv[1])

    #M1 = loadBinary(learnFileRoot)
    #plotTrainData(En, M, learnFileRoot, step)

#************************************
''' Open Learning Data '''
#************************************
def readLearnFile(learnFile):
    
    print(os.path.splitext(learnFile)[1])
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Learn data file not found \n' + '\033[0m')
        return

    learnFileRoot = os.path.splitext(learnFile)[0]

    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(np.array(M[:,1:]),np.s_[0:1],0)
    
    print("En:",En.shape)
    print("M:",M.shape)
    #print ("En:",En)
    #print("M:",M)
    return En, M, learnFileRoot

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
