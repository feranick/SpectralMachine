#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* Replicate training data with added noise
* Noise is a percentage of max in a spectra.
* For augmentation of data
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
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
    Ynorm = True
    YnormTo = 1

def main():
    if len(sys.argv) < 4:
        print(' Usage:\n  python3 AddRelativeNoisyData.py <learnData> <#additions> <%-offset>')
        print('  Data is by default normalized to 1\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    #newFile = os.path.splitext(sys.argv[1])[0] + '_n' + sys.argv[2] + '_offs' + sys.argv[3]
    newFile = os.path.splitext(sys.argv[1])[0] + '_n' + sys.argv[2] + '_oNpc' + sys.argv[3]
    learnFileExt = os.path.splitext(sys.argv[1])[1]

    '''
    if learnFileExt == ".h5":
        defParam.saveAsTxt = False
    else:
        defParam.saveAsTxt = True
    '''

    if len(sys.argv) == 5:
        defParam.addToFlatland = True
        newFile += '_back'
        print(' Adding ', sys.argv[2], 'sets with background random noise with offset:', sys.argv[3], '\n')
    else:
        print(' Adding', sys.argv[2], 'sets with random noise with offset:', sys.argv[3], '\n')

    En, M = readLearnFile(sys.argv[1])
    
    if defParam.Ynorm ==True:
        print(" Normalizing Learning Spectra to:",defParam.YnormTo)
        M = normalizeSpectra(M)
        newFile += '_norm1'

    if os.path.exists(newFile) == False:
        newTrain = np.append([0], En)
        newTrain = np.vstack((newTrain, M))
    else:
        newTrain = M

    for j in range(int(sys.argv[2])):
        newTrain = np.vstack((newTrain, scrambleNoise(M, float(sys.argv[3]))))

    if defParam.Ynorm ==True:
        print(" Normalizing Learning + Noisy Spectra to:",defParam.YnormTo,"\n")
        newTrain = normalizeSpectra(newTrain)

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
    M[:,1:] = np.multiply(M[:,1:],
        np.abs(np.multiply(np.random.uniform(-1,1, size=(1,M.shape[1]-1)),0.01*offset*np.amax(M[:,1:], axis = 0))))
    return M
'''
def scrambleNoiseOld(M, offset):
    from random import uniform
    for j in range(0, M.shape[0]):
        for i in range(1, M.shape[1]-1):
            if defParam.addToFlatland == False:
                #M[:,i] += 0.01*offset*uniform(-1,1)
                M[j,i] *= abs(0.01*offset*uniform(-1,1)*np.amax(M[j,:]))
            else:
                if M[:,i].any() == 0:
                    #M[:,i] += 0.01*offset*uniform(-1,1)
                    M[j,i] *= abs(0.01*offset*uniform(-1,1)*np.amax(M[j,:]))
    return M
'''
#************************************
''' Normalize '''
#************************************
def normalizeSpectra(M):
    for i in range(1,M.shape[0]):
        if(np.amin(M[i]) <= 0):
            M[i,1:] = M[i,1:] - np.amin(M[i,1:]) + 1e-8
        #M[i,1:] = np.multiply(M[i,1:], defParam.YnormTo/max(M[i][1:]))
    M[1:,1:] = np.multiply(M[1:,1:], np.array([float(defParam.YnormTo)/np.amax(M[1:,1:], axis = 1)]).T)
    return M

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
