#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* XmuDataMaker
* Adds spectra to single file for classification
* File must be in Xmu
* version: 20180301a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path
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
        enInit = 276
        enFin = 337
        enStep = 0.3
    else:
        enInit = sys.argv[2]
        enFin =  sys.argv[3]
        enStep = sys.argv[4]
        if len(sys.argv) < 6:
            threshold = 0
        else:
            threshold = sys.argv[5]

    if len(sys.argv) == 7:
        defParam.useMinForBoundary = True
    
    try:
        processMultiFile(sys.argv[1], enInit, enFin, enStep, threshold)
    except:
        usage()
    sys.exit(2)

#**********************************************
''' Open and process inividual files '''
#**********************************************
def processMultiFile(learnFile, enInit, enFin, enStep, threshold):
    index = 0
    success = False
    size = 0
    compound=[]
    learnFileRoot = os.path.splitext(learnFile)[0]
    summary_filename = learnFileRoot + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
    with open(summary_filename, "a") as sum_file:
        sum_file.write(str(datetime.now().strftime('Classification started: %Y-%m-%d %H:%M:%S'))+\
            ",enInit="+str(enInit)+",enFin="+str(enFin)+",enStep="+str(enStep)+\
            ",threshold="+str(threshold)+"\n")
        
    for ind, f in enumerate(sorted(os.listdir("."))):
        if (f != learnFile and os.path.splitext(f)[-1] == ".xmu"):
            try:
                index = compound.index(f.partition("_")[0])
            except:
                compound.append(f.partition("_")[0])
                index = len(compound)-1
            
            success = makeFile(f, learnFile, index, enInit, enFin, enStep, threshold)
            with open(summary_filename, "a") as sum_file:
                if success == True:
                    sum_file.write(str(index) + ',,,' + f +'\n')
                    size = size + 1
                else:
                    sum_file.write(str(index) + ',,NO,' + f +'\n')
    print('\n Energy scale: [', str(enInit),',',
            str(enFin), ']; Step:', str(enStep),
            '; Threshold:', str(threshold),'\n')

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
def makeFile(sampleFile, learnFile, param, enInit, enFin, enStep, threshold):
    learnFileRoot = os.path.splitext(learnFile)[0]
    print('\n Process file in class #: ' + str(param))
    try:
        with open(sampleFile, 'r') as f:
            #M = np.loadtxt(f, unpack = True, skiprows = 40)
            M = np.loadtxt(f, unpack = True)
        En = M[0]
        R = M[1]
            
        R[R<float(threshold)*np.amax(R)/100] = 0
        print(' Number of points in \"' + sampleFile + '\": ' + str(En.shape[0]))
        print(' Setting datapoints below ', threshold, '% of max (',str(np.amax(R)),')')
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
    print("Training file saved as text in:", learnFile)

    return True

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 XmuDataMaker.py <learnfile> <enInitial> <enFinal> <enStep> <threshold> \n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
