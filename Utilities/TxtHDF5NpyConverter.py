#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*******************************************************
* TxtHDF5NpyConverter
* Convert training files to/from text, hdf5, bin-npy
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
*******************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py

#**********************************************
''' main '''
#**********************************************
class defParam:
    saveAsTxt = False
def main():
    
    if len(sys.argv) < 2:
        usage()
        return
    try:
        print(" Opening earning file: "+sys.argv[1]+"\n")
        if os.path.splitext(sys.argv[1])[1] == ".npy":
            bin2text(sys.argv[1])
        else:
            text2bin(sys.argv[1])
    except:
        usage()
    sys.exit(2)

#**********************************************
''' Text to Binary format '''
#**********************************************
def text2bin(learnFile):
    learnFileRoot = os.path.splitext(learnFile)[0]
    if os.path.splitext(sys.argv[1])[1] == '.txt':
        with open(learnFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
    elif os.path.splitext(sys.argv[1])[1] == '.h5':
        with h5py.File(learnFile, 'r') as hf:
            M = hf["M"][:]
    else:
        print("File format not recognized")
        return

    file = open(learnFileRoot+".npy", 'ab')
    np.save(file, M, '%10.6f')
    print(" Learning file converted to\033[1m npy\033[0m: "+learnFileRoot+".npy\n")

#**********************************************
''' Binary to Text format '''
#**********************************************
def bin2text(learnFile):
    learnFileRoot = os.path.splitext(learnFile)[0]
    M = np.load(learnFile)
    if defParam.saveAsTxt == True:
        with open(learnFileRoot+".txt", 'ab') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
        print("Learning file converted to \033[1min\033[0m:", learnFileRoot+".txt\n")
    else:
        with h5py.File(learnFileRoot+'.h5', 'w') as hf:
            hf.create_dataset("M",  data=M)
        print(" Learning file converted to \033[1mhdf5\033[0m: "+learnFileRoot+".h5\n")

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 TxtHDF5NpyConverter.py <learnfile> \n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
