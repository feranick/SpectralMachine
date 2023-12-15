#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* NormLearnFile
* Normalize spectral intensity in Learning File
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py, time
#************************************
''' Main '''
#************************************
class defParam:
    saveAsTxt = False
    YnormTo = 1

def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3 NormLearnFile.py <learnData> <Ymax>\n')
        print(' Usage (norm intensity to 1):\n  python3 NormLearnFile.py <learnData>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    elif len(sys.argv) < 3:
        pass
    else:
        defParam.YnormTo = sys.argv[2]

    newFile = os.path.splitext(sys.argv[1])[0] + '_norm' + str(defParam.YnormTo)
    learnFileExt = os.path.splitext(sys.argv[1])[1]

    M = readLearnFile(sys.argv[1])
    M = normalizeSpectra(M)
    saveLearnFile(M, newFile)

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
    
    return M

#***************************************
''' Save new learning Data '''
#***************************************
def saveLearnFile(M, learnFile):
    if defParam.saveAsTxt == True:
        learnFile += '.txt'
        print(" Saving new training file (txt) in:", learnFile+"\n")
        with open(learnFile, 'ab') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    else:
        learnFile += '.h5'
        print(" Saving new training file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M)

#************************************
''' Normalize '''
#************************************
def normalizeSpectra(M):
    print(" Normalizing max spectral intensity to:",defParam.YnormTo,"\n")
    for i in range(1,M.shape[0]):
        #if(np.amin(M[i]) >= 0):
        M[i,1:] = M[i,1:] - np.amin(M[i,1:]) + 1e-8
        #M[i,1:] = np.multiply(M[i,1:], float(defParam.YnormTo)/max(M[i,1:]))
    M[1:,1:] = np.multiply(M[1:,1:], np.array([float(defParam.YnormTo)/np.amax(M[1:,1:], axis = 1)]).T)
    return M

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
