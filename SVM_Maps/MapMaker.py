#!/usr/bin/python
# -*- coding: utf-8 -*-
#*********************************************
#
# SVM_map_maker
# Adds spectra to map files
# version: 20160916c
#
# By: Nicola Ferralis <feranick@hotmail.com>
#
#**********************************************

import numpy as np
from sklearn import svm
import sys, os.path

#**********************************************
# main
#**********************************************

def main():
    try:
        makeFile(sys.argv[1], sys.argv[2], sys.argv[3])
    except:
        usage()
        sys.exit(2)

#**********************************************
# Learn and Predict
#**********************************************
def makeFile(mapFile, sampleFile, param):
    
    #**********************************************
    # Open and process training data
    #**********************************************

    with open(sampleFile, 'r') as f:
        En = np.loadtxt(f, unpack = True, usecols=range(0,1))
    
    with open(sampleFile, 'r') as f:
        R = np.loadtxt(f, unpack = True, usecols=range(1,2))

    if os.path.exists(mapFile):
        print('\n Adding spectra to ' + mapFile + '\n')
        newMap = np.append(float(param),R).reshape(1,-1)
    else:
        print('\n' + mapFile + ' does not exist. Creating...' + '\n')
        newMap = np.append([0], En)
        newMap = np.vstack((newMap, np.append(float(param),R)))

    with open(mapFile, 'ab') as f:
        np.savetxt(f, newMap, delimiter='\t', fmt='%10.6f')


####################################################################
''' Lists the program usage '''
####################################################################
def usage():
    print('\n Usage:')
    print('  python SVN_map_maker.py <mapfile> <spectrafile> <parameter> \n')

####################################################################
''' Main initialization routine '''
####################################################################
if __name__ == "__main__":
    sys.exit(main())
