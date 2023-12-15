#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* AddSpectraToLearnFile
* Adds spectra to Training File
* Spectra can be ASCII or Rruff
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os, os.path, h5py
from datetime import datetime, date

#**********************************************
''' main '''
#**********************************************
def main():
    try:
        if sys.argv[3] == ".":
            for ind, f in enumerate(sorted(os.listdir("."))):
                if f != sys.argv[1] and f != sys.argv[2] and os.path.splitext(f)[-1] == ".txt":
                    makeFile(sys.argv[1], sys.argv[2], f, sys.argv[4])
        else:
            makeFile(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    except:
        usage()
    sys.exit(2)

#**********************************************
''' Make Training file '''
#**********************************************
def makeFile(learnFile, sampleTag, sampleFile, param):
    #**********************************************
    ''' Open and process training data '''
    #**********************************************
    try:
        firstline = open(sampleFile).readline()
        with open(sampleFile, 'r') as f:
            if firstline[:7] == "##NAMES":
                En = np.loadtxt(f, unpack = True, usecols=range(0,1), delimiter = ',', skiprows = 10)
            else:
                En = np.loadtxt(f, unpack = True, usecols=range(0,1))
            print(' Number of points in \"' + sampleFile + '\": ' + str(En.shape[0]))
    except:
        print('\033[1m' + ' Sample data file not found \n' + '\033[0m')
        return

    with open(sampleFile, 'r') as f:
        if firstline[:7] == "##NAMES":
            R = np.loadtxt(f, unpack = True, usecols=range(1,2), delimiter = ',', skiprows = 10)
        else:
            R = np.loadtxt(f, unpack = True, usecols=range(1,2))

    if os.path.exists(learnFile):
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        elif os.path.splitext(learnFile)[1] == ".h5":
            with h5py.File(learnFile, 'r') as hf:
                M = hf["M"][:]
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
            
        sampleSize = M.shape[0]+1
        print(' Number of samples in \"' + learnFile + '\": ' + str(sampleSize))
        EnT = np.delete(np.array(M[0,:]),np.s_[0:1],0)
        if EnT.shape[0] == En.shape[0]:
            print(' Number of points in the map dataset: ' + str(EnT.shape[0]))
        else:
            print('\033[1m' + ' Mismatch in datapoints: ' + str(EnT.shape[0]) + '; sample = ' +  str(En.shape[0]) + '\033[0m')
            R = np.interp(EnT, En, R, left = 0, right = 0)
            print('\033[1m' + ' Mismatch corrected: datapoints in sample: ' + str(R.shape[0]) + '\033[0m')
        newTrain = np.append(float(param),R).reshape(1,-1)
    else:
        print('\n\033[1m' + ' Train data file not found. Creating...' + '\033[0m')
        newTrain = np.append([0], En)
        newTrain = np.vstack((newTrain, np.append(float(param),R)))
        sampleSize = 2

    saveLearnFile(newTrain, learnFile)
    
    if os.path.exists(sampleTag):
        learnFileInfo = sampleTag
        summary=""
    else:
        learnFileInfo = os.path.splitext(learnFile)[0] + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
        learnFileInfo = sampleTag
        print('\033[1m' + ' Train info file not found. Creating...' + '\033[0m')
        summary = str(datetime.now().strftime('Classification started: %Y-%m-%d %H:%M:%S'))+"\n"

    summary += str(param) + ',,,' + sampleFile +'\n'
        
    with open(learnFileInfo, 'ab') as f:
        f.write(summary.encode())

    print(' Added info to \"' + learnFileInfo + '\"\n')

#***************************************
''' Save new learning Data '''
#***************************************
def saveLearnFile(M, learnFile):
    if os.path.splitext(learnFile)[1] == '.txt':
        print(" Saving updated training file (txt) in:", learnFile+"\n")
        with open(learnFile, 'ab') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    elif os.path.splitext(sys.argv[1])[1] == '.h5':
        print(" Saving updated training file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M)
    elif os.path.splitext(sys.argv[1])[1] == '.npy':
        print(" Saving updated training file (npy) in: "+learnFile+"\n")
        with open(learnFile, 'ab') as f:
            np.save(f, M, '%10.6f')
    else:
        print(" Format of training file, "+learnFile+", not supported\n")

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 AddSpectraToLearnFile.py <trainingfile> <info_file> <spectrafile> <parameter>\n')
    print('  python3 AddSpectraToLearnFile.py <trainingfile> <info_file> . <parameter>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
