#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* TrainingDataMaker
* Adds spectra to Training File
* version: 20170522a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os

#**********************************************
''' main '''
#**********************************************

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
            En = np.loadtxt(f, unpack = True, usecols=range(0,1))
            print(' Number of points in \"' + sampleFile + '\": ' + str(En.shape[0]))
    except:
        print('\033[1m' + ' Sample data file not found \n' + '\033[0m')
        return

    with open(sampleFile, 'r') as f:
        R = np.loadtxt(f, unpack = True, usecols=range(1,2))

    if os.path.exists(trainFile):
        with open(trainFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
            sampleSize = M.shape[0]
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
        sampleSize = 1
    
    with open(trainFile, 'ab') as f:
        np.savetxt(f, newTrain, delimiter='\t', fmt='%10.6f')

    trainFileInfo = os.path.splitext(trainFile)[0] + "_README.txt"

    if os.path.exists(trainFileInfo):
        newTrainHeader = ""
    else:
        print('\033[1m' + ' Train info file not found. Creating...' + '\033[0m')
        newTrainHeader = "Tag \t File Name \t H:C \t row\n====================================================\n"

    newTrainInfo = newTrainHeader + sampleTag + "\t"+ sampleFile + "\tH:C=" + param + "\trow: " + str(sampleSize) +"\n"
        
    with open(trainFileInfo, 'ab') as f:
        f.write(newTrainInfo.encode())

    print(' Added info to \"' + trainFileInfo + '\"\n')


#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 TrainingDataMaker.py <trainingfile> <Tag> <spectrafile> <parameter>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
