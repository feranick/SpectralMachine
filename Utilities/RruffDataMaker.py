#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*************************************************
* RRuffDataMaker
* Adds spectra to single file for classification
* File must be in RRuFF
* version: v2024.10.07.1
* By: Nicola Ferralis <feranick@hotmail.com>
*************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py
from datetime import datetime, date

#**********************************************
''' main '''
#**********************************************
class defParam:
    saveAsTxt = False
    saveFormatClass = False

    # set boundaries intensities for when to
    # fill in in absence of data
    leftBoundary = 0
    rightBoundary = 0
    
    # set to True to set boundaries as the min
    # values for intensities when to
    # fill in in absence of data
    useMinForBoundary = True

def main():
    if len(sys.argv) < 5:
        enInit = 100
        enFin = 1500
        enStep = 0.5
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
    learnFileExt = os.path.splitext(learnFile)[1]
    
    if learnFileExt == ".txt" :
        defParam.saveAsTxt = True
    elif learnFileExt == ".h5" :
        defParam.saveAsTxt = False
    else:
        pass

    summary_filename = learnFileRoot + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.csv'))
    summary = str(datetime.now().strftime('Classification started: %Y-%m-%d %H:%M:%S'))+\
            ",enInit="+str(enInit)+",enFin="+str(enFin)+",enStep="+str(enStep)+\
            ",threshold="+str(threshold)+"\n"
    
    # Read, if exisiting, learnFile
    if os.path.exists(learnFile):
        print('\n\033[1m' + ' Train data file found. Opening...' + '\033[0m')
        EnT, M = readLearnFile(learnFile)
    else:
        print('\n\033[1m' + ' Train data file not found. Creating...' + '\033[0m')
        EnT = np.arange(float(enInit), float(enFin), float(enStep), dtype=float)
        M = np.append([0], EnT)

    # process sample data
    for ind, f in enumerate(sorted(os.listdir("."))):
        if (f != learnFile and os.path.splitext(f)[-1] == ".txt"):
            try:
                index = compound.index(f.partition("_")[0])
            except:
                compound.append(f.partition("_")[0])
                index = len(compound)-1
            
            success, M = makeFile(f, EnT, M, index, threshold)
            if success == True:
                summary += str(index) + ',,,' + f +'\n'
                size = size + 1
            else:
                summary += str(index) + ',,,' + f +'\n'

    print('\n Energy scale: [', str(enInit),',',
            str(enFin), ']; Step:', str(enStep),
            '; Threshold:', str(threshold),'\n')

    saveLearningFile(M, os.path.splitext(learnFile)[0])
    
    with open(summary_filename, "a") as sum_file:
        sum_file.write(summary)

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
def makeFile(sampleFile, EnT, M, param, threshold):
    print('\n Process file in class #: ' + str(param))
    try:
        with open(sampleFile, 'r') as f:
            En = np.loadtxt(f, unpack = True, usecols=range(0,1), delimiter = ',', skiprows = 10)
            if(En.size == 0):
                print('\n Empty file \n' )
                return False, M
        with open(sampleFile, 'r') as f:
            R = np.loadtxt(f, unpack = True, usecols=range(1,2), delimiter = ',', skiprows = 10)
        R[R<float(threshold)*np.amax(R)/100] = 0
        print(' Number of points in \"' + sampleFile + '\": ' + str(En.shape[0]))
        print(' Setting datapoints below ', threshold, '% of max (',str(np.amax(R)),')')
    except:
        print('\033[1m' + sampleFile + ' file not found \n' + '\033[0m')
        return False, M

    if EnT.shape[0] == En.shape[0]:
        print(' Number of points in the learning dataset: ' + str(EnT.shape[0]))
    else:
        print('\033[1m' + ' Mismatch in datapoints: ' + str(EnT.shape[0]) + '; sample = ' +  str(En.shape[0]) + '\033[0m')
        if defParam.useMinForBoundary == True:
            print(" Boundaries: Filling in with min values")
            #defParam.leftBoundary = R[0]
            #defParam.rightBoundary = R[R.shape[0]-1]
            defParam.leftBoundary = np.amin(R)
            defParam.rightBoundary = np.amin(R)
        else:
            print(" Boundaries: Filling in preset values")
        print("  Left:",defParam.leftBoundary,"; Right:",defParam.leftBoundary)
        
        R = np.interp(EnT, En, R, left = defParam.leftBoundary, right = defParam.rightBoundary)
        print('\033[1m' + ' Mismatch corrected: datapoints in sample: ' + str(R.shape[0]) + '\033[0m')

    M = np.vstack((M,np.append(float(param),R)))
    return True, M

#***************************************
''' Save learning file '''
#***************************************
def saveLearningFile(M, learnFileRoot):
    if defParam.saveAsTxt == True:
        learnFile = learnFileRoot+'.txt'
        print(" Saving new training file (txt) in:", learnFile+"\n")
        with open(learnFile, 'wb') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    else:
        learnFile = learnFileRoot+'.h5'
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M.astype(np.float64))
        print(" Saving new training file (hdf5) in: "+learnFile+"\n")

#************************************
''' Open Learning Data '''
#************************************
def readLearnFile(learnFile):
    print(" Opening learning file: "+learnFile+"\n")
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        elif os.path.splitext(learnFile)[1] == ".h5":
            with h5py.File(learnFile, 'r') as hf:
                M = hf["M"][:]
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
    except:
        print("\033[1m" + " Learning file not found \n" + "\033[0m")
        return

    En = M[0,1:]
    A = M[1:,1:]
    #Cl = M[1:,0]

    return En, M

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 RruffDataMaker.py <learnfile> <enInitial> <enFinal> <enStep> <threshold> \n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
