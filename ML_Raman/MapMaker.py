#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
*********************************************
*
* MapMakerSVM
* Adds spectra to map files
* version: 20160919f
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''

print(__doc__)

import numpy as np
from sklearn import svm
import sys, os.path

#**********************************************
''' main '''
#**********************************************

def main():
    #try:
        makeFile(sys.argv[1], sys.argv[2], sys.argv[3])
            #except:
            #usage()
#sys.exit(2)

#**********************************************
''' Make Map file '''
#**********************************************
def makeFile(mapFile, param, sampleFile):
    
    #**********************************************
    ''' Open and process training data '''
    #**********************************************
    
    try:
        with open(sampleFile, 'r') as f:
            En = np.loadtxt(f, unpack = True, usecols=range(0,1))
            print(' Number of points in the sample dataset: ' + str(En.shape[0]))
    except:
        print('\033[1m' + ' Sample data file not found \n' + '\033[0m')
        return

    with open(sampleFile, 'r') as f:
        R = np.loadtxt(f, unpack = True, usecols=range(1,2))

    try:
        with open(mapFile, 'r') as f:
            if len(f.readline().split('\t'))-1 == En.shape[0]:
                print(' Number of points in the map dataset: ' + str(len(f.readline().split('\t'))-1))
            else:
                print('\033[1m' + ' Mismatch in datapoints: map: ' + str(len(f.readline().split('\t'))-1) + '; sample = ' +  str(En.shape[0]) + '\033[0m' + '\n')
                return
    except:
        print('\033[1m' + ' Map data file not found \n' + '\033[0m')
        return


    if os.path.exists(mapFile):
        print('\n Adding spectra to ' + mapFile + '\n')
        newMap = np.append(float(param),R).reshape(1,-1)
    else:
        print('\n' + mapFile + ' does not exist. Creating...' + '\n')
        newMap = np.append([0], En)
        newMap = np.vstack((newMap, np.append(float(param),R)))

    with open(mapFile, 'ab') as f:
        np.savetxt(f, newMap, delimiter='\t', fmt='%10.6f')


#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:')
    print('  python MapMakerSVM.py <mapfile> <spectrafile> <parameter> \n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
