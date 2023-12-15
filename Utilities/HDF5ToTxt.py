#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* HDF5ToTxt
* Convert HDF5 learning data to Txt
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
*********************************************
'''
print(__doc__)
import numpy as np
import sys, os.path, h5py

#************************************
''' Main '''
#************************************
def main():

    if len(sys.argv) < 2:
        print(' Usage:\n  python3 HDF5ToTxt.py <Learning File in HDF5 format>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        file =  sys.argv[1]

    convertLearnFile(file)

#************************************
''' Read Learning file - HDF5'''
#************************************
def convertLearnFile(learnFile):
    learnFileTxt = os.path.splitext(learnFile)[0]+".txt"
    
    try:
        print(" Opening training file (hdf5): "+learnFile)
        with h5py.File(learnFile, 'r') as hf:
            M = hf["M"][:]
    except:
        print(" Training file (hdf5) not found\n")
        return
    
    if os.path.exists(learnFileTxt) == False:
        with open(learnFileTxt, 'wb') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
        print(" Training file saved as Txt in: "+learnFileTxt+"\n")
    else:
        print(" Training file as Txt already exists: "+learnFileTxt+"\n")
    
#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
