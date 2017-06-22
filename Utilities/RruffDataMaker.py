#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* RRuffDataMaker
* Adds spectra to single file for classification
* File must be in RRuFF
* version: 20170622a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, glob
from datetime import datetime, date

#**********************************************
''' main '''
#**********************************************

class defParam:
    saveFormatClass = False

    # set boundaries intensities for when to
    # fill in in absence of data
    leftBoundary = 0
    rightBoundary = 0
    
    # set to True to set boundaries as the min
    # values for intensities when to
    # fill in in absence of data
    useMinForBoundary = False

def main():
    if len(sys.argv) < 5:
        enInit = 100
        enFin = 1500
        enStep = 0.5
    else:
        enInit = sys.argv[2]
        enFin =  sys.argv[3]
        enStep = sys.argv[4]

    if len(sys.argv) == 6:
        defParam.useMinForBoundary = True
    
    try:
        processMultiFile(sys.argv[1], enInit, enFin, enStep)
    except:
        usage()
    sys.exit(2)

#**********************************************
''' Open and process inividual files '''
#**********************************************
def processMultiFile(learnFile, enInit, enFin, enStep):
    index = 0
    success = False
    size = 0
    compound=[]
    learnFileRoot = os.path.splitext(learnFile)[0]
    summary_filename = learnFileRoot + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.log'))
    with open(summary_filename, "a") as sum_file:
        sum_file.write(str(datetime.now().strftime('Classification started: %Y-%m-%d %H:%M:%S\n')))
    
    for f in glob.glob('*.txt'):
        if (f != learnFile):
            
            try:
                index = compound.index(f.partition("_")[0])
            except:
                compound.append(f.partition("_")[0])
                index = len(compound)-1
            
            success = makeFile(f, learnFile, index, enInit, enFin, enStep)
            with open(summary_filename, "a") as sum_file:
                if success == True:
                    sum_file.write(str(index) + '\t\t' + f +'\n')
                    size = size + 1
                else:
                    sum_file.write(str(index) + '\tNO\t' + f +'\n')
    print('\n Energy scale: [' + str(enInit) + ', ' + str(enFin) + '] Step: ' + str(enStep) + '\n')

    Cl2 = np.zeros((size, size))
    for i in range(size):
        np.put(Cl2[i], i, 1)

    if defParam.saveFormatClass == True:
        tfclass_filename = learnFileRoot + '.tfclass'
        print(' Saving class file...\n')
        with open(tfclass_filename, 'ab') as f:
            np.savetxt(f, Cl2, delimiter='\t', fmt='%10.6f')

#**********************************************
''' Add data to Training file '''
#**********************************************
def makeFile(sampleFile, learnFile, param, enInit, enFin, enStep):
    print('\n Process file in class #: ' + str(param))
    try:
        with open(sampleFile, 'r') as f:
            En = np.loadtxt(f, unpack = True, usecols=range(0,1), delimiter = ',', skiprows = 10)
            if(En.size == 0):
                print('\n Empty file \n' )
                return False
        with open(sampleFile, 'r') as f:
            R = np.loadtxt(f, unpack = True, usecols=range(1,2), delimiter = ',', skiprows = 10)
        print(' Number of points in \"' + sampleFile + '\": ' + str(En.shape[0]))
    except:
        print('\033[1m' + sampleFile + ' file not found \n' + '\033[0m')
        return False

    EnT = np.arange(float(enInit), float(enFin), float(enStep), dtype=np.float)
    if EnT.shape[0] == En.shape[0]:
        print(' Number of points in the learning dataset: ' + str(EnT.shape[0]))
    else:
        print('\033[1m' + ' Mismatch in datapoints: ' + str(EnT.shape[0]) + '; sample = ' +  str(En.shape[0]) + '\033[0m')
        if defParam.useMinForBoundary == True:
            print(" Boundaries: Filling in with min values")
            defParam.leftBoundary = R[0]
            defParam.rightBoundary = R[R.shape[0]-1]
        else:
            print(" Boundaries: Filling in preset values")
        print("  Left:",defParam.leftBoundary,"; Right:",defParam.leftBoundary)
        
        R = np.interp(EnT, En, R, left = defParam.leftBoundary, right = defParam.rightBoundary)
        print('\033[1m' + ' Mismatch corrected: datapoints in sample: ' + str(R.shape[0]) + '\033[0m')

    if os.path.exists(learnFile):
        newTrain = np.append(float(param),R).reshape(1,-1)
    else:
        print('\n\033[1m' + ' Train data file not found. Creating...' + '\033[0m')
        newTrain = np.append([0], EnT)
        print(' Added spectra to \"' + learnFile + '\"\n')
        newTrain = np.vstack((newTrain, np.append(float(param),R)))

    with open(learnFile, 'ab') as f:
        np.savetxt(f, newTrain, delimiter='\t', fmt='%10.6f')

    return True

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 RruffDataMaker.py <learnfile> <enInitial> <enFinal> <enStep> \n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
