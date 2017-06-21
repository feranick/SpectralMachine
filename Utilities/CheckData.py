#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* Check whether a training file has data with
*  intensity higher than specified value
*
* version: 20170621a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)


import numpy as np
import sys, os.path, getopt, glob, csv

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
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
