#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************
* SpectraExtractorSVM.py
* Extract spectra of specific phases
* version: v2023.12.15-1
* By: Nicola Ferralis <feranick@hotmail.com>
**********************************************

'''
print(__doc__)

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import sys, os.path

#**********************************************
''' Options '''
#**********************************************
phaseColumn = 8
selPhase = 6
labelColumn = 2

#**********************************************
''' Main '''
#**********************************************
def main():
    try:
        spectraExtractor(sys.argv[1], sys.argv[2], sys.argv[3])
    except:
        usage()
        sys.exit(2)

#**********************************************
''' Spectra Extractor '''
#**********************************************
def spectraExtractor(mapFile, clustFile, newMapFile):

    #**********************************************
    ''' Read input data files '''
    #**********************************************

    print(' Reading data and cluster files... \n')

    try:
        with open(mapFile, 'r') as f:
            En = np.array(f.readline().split(), dtype=np.dtype(float))
            A = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Map data file not found \n' + '\033[0m')
        return
    
    A = np.delete(A, np.s_[0:2], 1)
    print(' Shape map: ' + str(A.shape))

    try:
        with open(clustFile, 'r') as f:
            Cl = np.loadtxt(f, unpack =False, skiprows=1, usecols = range(phaseColumn-1,phaseColumn))
    except:
        print('\033[1m' + '\n Cluster data file not found \n' + '\033[0m')
        return

    with open(clustFile, 'r') as f:
        L = np.loadtxt(f, unpack =False, skiprows=1, usecols = range(labelColumn-1,labelColumn))

    print(' Shape cluster vector: ' + str(Cl.shape))
    print(' Shape label vector: ' + str(L.shape))

    #**********************************************
    ''' Create new map file '''
    #**********************************************
    print('\n Creating new map file: ' + newMapFile)
    phaseMap = np.append([0], En)

    for i in range(0,A.shape[0]):
        if Cl[i] == selPhase:
            temp = np.append(L[i], A[i,:])
            phaseMap = np.vstack((phaseMap, temp))

    print(' Shape new map: ' + str(phaseMap.shape) + '\n')

    np.savetxt(newMapFile, phaseMap, delimiter='\t', fmt='%10.6f')

#**********************************************
''' Lists the program usage '''
#**********************************************
def usage():
    print('\n Usage:\n')
    print('  python3 SpectraExtractSVM.py <mapfile> <cluster file> <new map file>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#**********************************************
''' Main initialization routine '''
#**********************************************
if __name__ == "__main__":
    sys.exit(main())

