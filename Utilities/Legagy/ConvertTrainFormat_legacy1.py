#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Convert train data to binary or text file
* version: 20180203b
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
        print(' Usage:\n  python3 ConvertTrainFormat.py <learnData>\n')
        print(' If <learnData.txt> is in text format -> binary npy')
        print(' If <learnData.npy> is in binary format -> text txt\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    readLearnFile(sys.argv[1])

#************************************
''' Open Learning Data '''
#************************************
def readLearnFile(learnFile):
    try:
        learnFileRoot = os.path.splitext(learnFile)[0]
        
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
            print("M:",M.shape)
            with open(learnFileRoot+".txt", 'ab') as f:
                np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
            print("Training file saved as text in:", learnFileRoot+".txt")
    
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
            print("M:",M.shape)
            np.save(learnFileRoot, M, '%10.6f')
            print("Training file saved as binary in:", learnFileRoot+"npy")
    
    except:
        print('\033[1m' + ' Learn data file not found \n' + '\033[0m')
        return

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
