#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* PCADataMaker
* Adds spectra to single file for PCA
* version: v2023.12.15-1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, glob

#**********************************************
''' main '''
#**********************************************

def main():
    if len(sys.argv) < 3:
        param = 0.0
    else:
        param = sys.argv[2]

    try:
        processMultiFile(sys.argv[1], param)
    except:
        usage()
    sys.exit(2)

#**********************************************
''' Open and process inividual files '''
#**********************************************
def processMultiFile(pcaFile, param):
    for f in glob.glob('*.txt'):
        if (f != pcaFile):
            makeFile(f, pcaFile, param)
    
#**********************************************
''' Add data to PCA file '''
#**********************************************
def makeFile(sampleFile, pcaFile, param):
    try:
        with open(sampleFile, 'r') as f:
            En = np.loadtxt(f, unpack = True, usecols=range(0,1))
            print(' Number of points in \"' + sampleFile + '\": ' + str(En.shape[0]))
    except:
        print('\033[1m' + ' Sample data file not found \n' + '\033[0m')
        return

    with open(sampleFile, 'r') as f:
        R = np.loadtxt(f, unpack = True, usecols=range(1,2))

    if os.path.exists(pcaFile):
        with open(pcaFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
            EnT = np.delete(np.array(M[0,:]),np.s_[0:1],0)
            if EnT.shape[0] == En.shape[0]:
                print(' Number of points in the pca dataset: ' + str(EnT.shape[0]))
            else:
                print('\033[1m' + ' Mismatch in datapoints: ' + str(EnT.shape[0]) + '; sample = ' +  str(En.shape[0]) + '\033[0m')
                R = np.interp(EnT, En, R, left = 0, right = 0)
                print('\033[1m' + ' Mismatch corrected: datapoints in sample: ' + str(R.shape[0]) + '\033[0m')
            print('\n Added spectra to \"' + pcaFile + '\"\n')
            newTrain = np.append(float(param),R).reshape(1,-1)
    else:
        print('\n\033[1m' + ' Train data file not found. Creating...' + '\033[0m')
        newTrain = np.append([0], En)
        print(' Added spectra to \"' + pcaFile + '\"\n')
        newTrain = np.vstack((newTrain, np.append(float(param),R)))

    with open(pcaFile, 'ab') as f:
        np.savetxt(f, newTrain, delimiter='\t', fmt='%10.6f')

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 PCADataMaker.py <pcafile> <parameter>\n')
    print('  Note: a default <parameter> = 0 is used if not declared\n')
    print('  <parameter> is useful for differentiate between different')
    print('  datasets within PCA. \n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
