#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Convert train data to binary file
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
        print(' Usage:\n  python3 SaveBinary.py <learnData> <step>\n')
        print(' Usage (full set):\n  python3 SaveBinary.py <learnData>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    readLearnFile(sys.argv[1])

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

    learnFileRoot = os.path.splitext(learnFile)[0]
    print("M:",M.shape)
    np.save(learnFileRoot, M, '%10.6f')
    print("Binary Traning file save in:",learnFileRoot+".npy")

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
