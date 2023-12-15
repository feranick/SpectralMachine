#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* Replicate training data with added noise
* Noise is a percentage of max in a spectra.
* spectra are also shifted along the x axis
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
    randOffset = False

def main():
    if len(sys.argv) < 5:
        print(' Usage:\n  python3 AddRelativeHorNoisyData.py <learnData> <#additions> <%-noise offset> <Hor-offset>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    newFile = os.path.splitext(sys.argv[1])[0] + '_n' + sys.argv[2] + '_oNpc' + sys.argv[3] + '_oH' + sys.argv[4]
    
    if len(sys.argv) == 6:
        defParam.addToFlatland = True
        newFile += '_back'
        print(' Adding ', sys.argv[2], 'sets with background random noise with offset:', sys.argv[3],
              ', horizontal offset within', sys.argv[4],'\n')
    else:
        print(' Adding', sys.argv[2], 'sets with random noise with offset:', sys.argv[3],
              ', horizontal offset within', sys.argv[4],'\n')

    En, M = readLearnFile(sys.argv[1])

    if os.path.exists(newFile) == False:
        newTrain = np.append([0], En)
        newTrain = np.vstack((newTrain, M))
    else:
        newTrain = M

    for j in range(int(sys.argv[2])):
        newTrain = np.vstack((newTrain,
                    scrambleNoise(horizontalOffset(En, M, float(sys.argv[3])),
                    float(sys.argv[3]))))

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
    from random import uniform
    for j in range(0, M.shape[0]):
        for i in range(1, M.shape[1]):
            if defParam.addToFlatland == False:
                M[j,i] += offset*uniform(-1,1)*np.amax(M[j,:])
            else:
                if M[j,i].any() == 0:
                    M[j,i] += offset*uniform(-1,1)*np.amax(M[j,:])
    return M
    '''

#*******************************************
''' Introduce Horizontal Offset in Data '''
#*******************************************
def horizontalOffset(En, M, offset, rand):
    newM = np.copy(M)
    for i in range(0, M.shape[0]):
        if defParam.randOffset:
            newEn = np.add(En, offset*np.random.uniform(-1,1))
        else:
            newEn = np.add(En, offset)
        newM[i,1:] = np.interp(En, newEn, M[i,1:], left = 0, right = 0)
    return newM
    '''
    newM = np.zeros(M.shape)
    newM[:,0] = M[:,0]
    newEn = np.zeros(En.shape)
    
    for i in range(0, M.shape[0]):
        if rand is True:
            from random import uniform
            newEn = En + offset*uniform(-1,1)
        else:
            newEn = En + offset
        newM[i,1:] = np.interp(En, newEn, M[i,1:], left = 0, right = 0)
    
    return newM
    '''
#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
