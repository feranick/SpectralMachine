#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*************************************************
*
* Replicate training data with horizontal offset
* For augmentation of data
*
* version: 20170807a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
*************************************************
'''
print(__doc__)


import numpy as np
import sys, os.path, getopt, glob, csv

def main():
    if(len(sys.argv)<3):
        print(' Usage:\n  python3 AddHorizontalOffset.py <learnData> <offset>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, M = readLearnFile(sys.argv[1])
    newFile = os.path.splitext(sys.argv[1])[0] + '_Hoffs' + sys.argv[2] + '.txt'

    if os.path.exists(newFile) == False:
        newTrain = np.append([0], En)
        newTrain = np.vstack((newTrain, M))
    else:
        newTrain = M

    print(' Adding sets with horizontal offset:', sys.argv[2], '\n')

    newTrain = np.vstack((newTrain, horizontalOffset(En, M, float(sys.argv[2]), True)))

    with open(newFile, 'ab') as f:
        np.savetxt(f, newTrain, delimiter='\t', fmt='%10.6f')

    print(' New training file saved: ', newFile, '\n')

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
''' Introduce Noise in Data '''
#************************************
def horizontalOffset(En, M, offset, rand):
    if rand is True:
        from random import uniform
        newEn = En + offset*uniform(-1,1)
    else:
        newEn = En + offset

    newM = np.zeros(M.shape)
    newM[:,0] = M[:,0]

    print(newM)

    for i in range(0, M.shape[0]):
        newM[i,1:] = np.interp(En, newEn, M[i,1:], left = 0, right = 0)
        print(i)
        print(En[50], M[0,50])
        print(newEn[50], newM[0,50])

    print("M", M)
    print("newM:", newM)
    print("En", En)
    print("newEn:", newEn)
    
    return newM


#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
