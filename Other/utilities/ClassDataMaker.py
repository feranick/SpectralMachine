#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
*********************************************
*
* ClassDataMaker
* Adds spectra to single file for classification
* version: 20170306a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, glob

#**********************************************
''' main '''
#**********************************************

def main():
    try:
        processMultiFile(sys.argv[1])
    except:
        usage()
    sys.exit(2)

#**********************************************
''' Open and process inividual files '''
#**********************************************
def processMultiFile(pcaFile):
    index = 0
    for f in glob.glob('*.txt'):
        if (f != pcaFile):
            makeFile(f, pcaFile, index)
            index = index + 1
    
#**********************************************
''' Add data to Training file '''
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
                R = np.interp(EnT, En, R)
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
    print('\n Usage:')
    print('  python ClassDataMaker.py <pcafile>\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
