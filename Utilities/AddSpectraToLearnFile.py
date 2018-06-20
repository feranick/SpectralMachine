#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* AddSpectraToLearnFile
* Adds spectra to Training File
* version: 20180619c
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os, h5py
from datetime import datetime, date

#**********************************************
''' main '''
#**********************************************
class defParam:
    isRruff = False

def main():
    try:
        makeFile(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except:
        usage()
    sys.exit(2)

#**********************************************
''' Make Training file '''
#**********************************************
def makeFile(trainFile, sampleTag, sampleFile, param):
    #**********************************************
    ''' Open and process training data '''
    #**********************************************
    try:
        with open(sampleFile, 'r') as f:
            if defParam.isRruff:
                En = np.loadtxt(f, unpack = True, usecols=range(0,1), delimiter = ',', skiprows = 10)
            else:
                En = np.loadtxt(f, unpack = True, usecols=range(0,1))
            print(' Number of points in \"' + sampleFile + '\": ' + str(En.shape[0]))
    except:
        print('\033[1m' + ' Sample data file not found \n' + '\033[0m')
        return

    with open(sampleFile, 'r') as f:
        if defParam.isRruff:
            R = np.loadtxt(f, unpack = True, usecols=range(1,2), delimiter = ',', skiprows = 10)
        else:
            R = np.loadtxt(f, unpack = True, usecols=range(1,2))

    if os.path.exists(trainFile):
        with open(trainFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
            sampleSize = M.shape[0]+1
            print(' Number of samples in \"' + trainFile + '\": ' + str(sampleSize))
            EnT = np.delete(np.array(M[0,:]),np.s_[0:1],0)
            if EnT.shape[0] == En.shape[0]:
                print(' Number of points in the map dataset: ' + str(EnT.shape[0]))
            else:
                print('\033[1m' + ' Mismatch in datapoints: ' + str(EnT.shape[0]) + '; sample = ' +  str(En.shape[0]) + '\033[0m')
                R = np.interp(EnT, En, R, left = 0, right = 0)
                print('\033[1m' + ' Mismatch corrected: datapoints in sample: ' + str(R.shape[0]) + '\033[0m')
            print('\n Added spectra to: \"' + trainFile + '\"')
            newTrain = np.append(float(param),R).reshape(1,-1)
    else:
        print('\n\033[1m' + ' Train data file not found. Creating...' + '\033[0m')
        newTrain = np.append([0], En)
        print(' Added spectra to: \"' + trainFile + '\"')
        newTrain = np.vstack((newTrain, np.append(float(param),R)))
        sampleSize = 2
    
    with open(trainFile, 'ab') as f:
        np.savetxt(f, newTrain, delimiter='\t', fmt='%10.6f')

    if os.path.exists(sampleTag):
        trainFileInfo = sampleTag
        summary=""
    else:
        trainFileInfo = os.path.splitext(trainFile)[0] + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
        trainFileInfo = sampleTag
        print('\033[1m' + ' Train info file not found. Creating...' + '\033[0m')
        summary = str(datetime.now().strftime('Classification started: %Y-%m-%d %H:%M:%S'))+"\n"

    summary += str(param) + ',,,' + sampleFile +'\n'
        
    with open(trainFileInfo, 'ab') as f:
        f.write(summary.encode())

    print(' Added info to \"' + trainFileInfo + '\"\n')

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 TrainingDataMaker.py <trainingfile> <info_file> <spectrafile> <parameter>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
