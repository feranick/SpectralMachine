#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
**********************************************
*
* SpectraExtractorSVM.py
* Extract spectra of specific phases
* version: 20160916e
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
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
    with open(mapFile, 'r') as f:
        En = np.array(f.readline().split(), dtype=np.dtype(float))

    f = open(mapFile, 'r')
    A = np.loadtxt(f, unpack =False, skiprows=1)
    A = np.delete(A, np.s_[0:2], 1)
    f.close()
    print(' Shape map: ' + str(A.shape))

    f = open(clustFile, 'r')
    Cl = np.loadtxt(f, unpack =False, skiprows=1, usecols = range(phaseColumn-1,phaseColumn))
    f.close()
    print(' Shape cluster vector: ' + str(Cl.shape))

    f = open(clustFile, 'r')
    L = np.loadtxt(f, unpack =False, skiprows=1, usecols = range(labelColumn-1,labelColumn))
    f.close()
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
    print('\n Usage:')
    print('  python SpectraExtractSVM.py <mapfile> <cluster file> <new map file>\n')

#**********************************************
''' Main initialization routine '''
#**********************************************
if __name__ == "__main__":
    sys.exit(main())

