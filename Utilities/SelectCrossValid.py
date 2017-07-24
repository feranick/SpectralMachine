#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*************************************************
*
* Select Cross Validation Dataset from log file
*
* version: 20170724b
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

    for i in range(1,M.shape[0]):
        if M[i] != 0:
            L = np.append(L,[i])

    cvSize = L.shape[0]*100/M.shape[0]

    print("\n Size of training set:", str(M.shape[0]),
          "\n Size of testing set: ",str(L.shape[0]),
          " ({:.2f}%)\n".format(cvSize))

    L = L.reshape(-1,1)

    print(' Saving <list_file> in:', listFile, '\n')
    with open(listFile, 'ab') as f:
        np.savetxt(f, L, delimiter='\t', fmt='%1d')
    print(' To create cross validation data set from learning set run:\n',
          '  MakeCrossValidSet.py <learningFile> <list_file> \n')

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
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
