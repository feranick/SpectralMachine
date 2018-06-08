#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* ReadHDF5
* Read HDF5 learning data
*
* version: 20180608a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''

import numpy as np
import sys, os.path, h5py


#************************************
''' Main '''
#************************************
def main():

    if len(sys.argv) < 2:
        print(' Usage:\n  python3 ReadHDF5.py <Learning File in HDF5 format>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        file =  sys.argv[1]

    readLearnFile(file)

#************************************
''' Read Learning file - HDF5'''
#************************************
def readLearnFile(learnFile):

    with h5py.File(learnFile, 'r') as hf:
        En = hf["En"][:]
        A = hf["A"][:]
        Cl = hf["Cl"][:].astype('<U4')
        
    print("En: ", En)
    print("A: ",A)
    print("Cl: ",Cl)
    return 0
#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
