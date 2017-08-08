#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* Replicate training data with vertical offset
* For augmentation of data
*
* version: 20170807b
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)


import numpy as np
import sys, os.path, getopt, glob, csv

def main():
    if(len(sys.argv)<4):
        print(' Usage:\n  python3 AddVerticalOffset.py <learnData> <#additions> <offset>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, M = readLearnFile(sys.argv[1])
    newFile = os.path.splitext(sys.argv[1])[0] + '_num' + sys.argv[2]+ '_Voffs' + sys.argv[3] + '.txt'

    if os.path.exists(newFile) == False:
        newTrain = np.append([0], En)
        newTrain = np.vstack((newTrain, M))
    else:
        newTrain = M

    print(' Adding', sys.argv[2],'sets with vertical offset:', sys.argv[3], '\n')

    for j in range(int(sys.argv[2])):
        newTrain = np.vstack((newTrain, verticalOffset(M, float(sys.argv[3]))))

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
def verticalOffset(M, offset):    
    for i in range(1, M.shape[1]-1):
        M[:,i] += offset
    
    return M

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
