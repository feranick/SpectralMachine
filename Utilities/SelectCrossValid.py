#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*************************************************
*
* Select Cross Validation Dataset from log file
*
* version: 20170724a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
*************************************************
'''
print(__doc__)


import numpy as np
import sys, os.path, getopt, glob

def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 SelectCrossValid.py <index csv file>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    listFile = os.path.splitext(sys.argv[1])[0] + '_list.txt'

    if os.path.exists(listFile) == True:
        print(" Training or Cross validation files exist. Exiting.\n")
        return

    M = readIndexFile(sys.argv[1])

    L = np.zeros(0)
    print(M)
    
    for i in range(1,M.shape[0]):
        if M[i] != 0:
            print(i)
            print(M[i])
            L = np.append(L,[i])


    L = L.reshape(-1,1)
    

    print(' Saving new cross validation file in:', listFile, '\n')
    with open(listFile, 'ab') as f:
        np.savetxt(f, L, delimiter='\t', fmt='%1d')
    print(' Done!\n')

#************************************
''' Open Index File '''
#************************************
def readIndexFile(File):
    try:
        csv = np.genfromtxt (File, delimiter=",")
        M = np.nan_to_num(csv[:,1])
    except:
        print('\033[1m' + ' Index data file not found \n' + '\033[0m')
        return

    return M

#************************************
''' Open list '''
#************************************
def readList(File):
    try:
        with open(File, 'r') as f:
            L = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' List data file not found \n' + '\033[0m')
        return
    
    return L

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
