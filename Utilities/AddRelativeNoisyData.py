#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* Replicate training data with added noise
* Noise is a percentage of max in a spectra.
* For augmentation of data
*
* version: 20180615a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, h5py
#************************************
''' Main '''
#************************************
class defParam:
    saveAsTxt = False
    addToFlatland = False

def main():
    if len(sys.argv) < 4:
        print(' Usage:\n  python3 AddRelativeNoisyData.py <learnData> <#additions> <%-offset>')
        print('  Data is NOT normalized.\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    newFile = os.path.splitext(sys.argv[1])[0] + '_n' + sys.argv[2] + '_oNpc' + sys.argv[3]
    
    if len(sys.argv) == 5:
        defParam.addToFlatland = True
        newFile += '_back'
        print(' Adding ', sys.argv[2], 'sets with background random noise with offset:', sys.argv[3], '\n')
    else:
        print(' Adding', sys.argv[2], 'sets with random noise with offset:', sys.argv[3], '\n')

    En, M = readLearnFile(sys.argv[1])

    if os.path.exists(newFile) == False:
        newTrain = np.append([0], En)
        newTrain = np.vstack((newTrain, M))
    else:
        newTrain = M

    for j in range(int(sys.argv[2])):
        newTrain = np.vstack((newTrain, scrambleNoise(M, float(sys.argv[3]))))

    saveLearnFile(newTrain, newFile)

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
    A = M[1:,:]
    #Cl = M[1:,0]
    return En, A

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
''' Introduce Noise in Data '''
#************************************
def scrambleNoise(M, offset):
    from random import uniform
    for j in range(0, M.shape[0]):
        for i in range(1, M.shape[1]):
            if defParam.addToFlatland == False:
                M[j,i] += offset*uniform(-1,1)*np.amax(M[j,:])
            else:
                if M[j,i].any() == 0:
                    M[j,i] += offset*uniform(-1,1)*np.amax(M[j,:])
    return M

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
