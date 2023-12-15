#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* Remove Column
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py, pickle
from random import uniform
from bisect import bisect_left

#************************************
# Parameters definition
#************************************
class dP:
    saveAsTxt = True
    precData = 2
#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 RemoveColumn.py <learnFile> <column>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    M = readLearnFile(sys.argv[1])
    if int(sys.argv[2]) != 0 and int(sys.argv[2]) <= M.shape[1]:
        Mdel = np.delete(M,sys.argv[2],1)
        rootFile = os.path.splitext(sys.argv[1])[0]
        saveLearnFile(Mdel, rootFile+"_exclCol"+sys.argv[2])
    else:
        return

#************************************
# Open Learning Data
#************************************
def readLearnFile(learnFile):
    print("\n  Opening learning file: "+learnFile+"\n")
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
        print("\033[1m" + "  Learning file not found \n" + "\033[0m")
        return

    print (M)
    return M

#***************************************
# Save new learning Data
#***************************************
def saveLearnFile(M, learnFile):
    if dP.saveAsTxt == True:
        learnFile += '.txt'
        with open(learnFile, 'ab') as f:
                 np.savetxt(f, M, delimiter='\t', fmt="%10.{0}f".format(dP.precData))
        print(" Saving new training file (txt) in:", learnFile+"\n")
    else:
        learnFile += '.h5'
        print(" Saving new training file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M)

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
