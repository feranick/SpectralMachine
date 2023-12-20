#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************************
* Data augmentation via High Freqency data average
* version: v2023.12.20.1
* By: Nicola Ferralis <feranick@hotmail.com>
*********************************************************
'''
print(__doc__)

import numpy as np
import sys, os, csv, h5py

class defParam:
    saveAsTxt = False

def main():
    if(len(sys.argv)<3):
        print(' Usage:\n  python3 AddMixHighFreqMaker.py <learnData> <HighFreq threshold>')
        print('  python3 AddMixHighFreqMaker.py <learnData> <HighFreq threshold> <iterations>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
        
    if(len(sys.argv)<4):
        iter = 1
    else:
        iter = int(sys.argv[3])
    
    M, En, A, Cl = readLearnFile(sys.argv[1])

    newTrainFile = os.path.splitext(sys.argv[1])[0] + '_av-hf' + sys.argv[2] + '_iter' + str(iter)

    if defParam.saveAsTxt == True:
        newTrainFile = newTrainFile + '.txt'
    else:
        newTrainFile = newTrainFile + '.h5'

    if os.path.exists(newTrainFile) == True:
        print(" Augmented training file with similar confuiguration exists. Exiting.\n")
        return

    M = selectHFdata(M,int(sys.argv[2]), iter)
    writeFile(newTrainFile, M)
    print('\n Writing new training file: ', newTrainFile)
    print('\n Done!\n')
    
#************************************
''' Mix high frequency datasets '''
#************************************
def selectHFdata(M, HFthreshold, iter):
    print(" Creating new datapoints by averaging members of classes with more than {0:.0f} datapoints...".format(HFthreshold))
    for h in range(iter):
        A = M[1:,1:]
        Cl = M[1:,0]
    
        uHFCl = np.array([])
        uniCl = np.unique(Cl).astype(int)
        for x in enumerate(uniCl):
            if np.count_nonzero(Cl==uniCl[x[0]]) >= HFthreshold:
                uHFCl = np.append(uHFCl, uniCl[x[0]])
        
        for i in uHFCl:
            list = np.where(np.in1d(Cl,i) == True)[0]
            rlist = list
            np.random.shuffle(rlist)
        
            for l in range(list.shape[0]):
                A_tmp = np.vstack((A[list[l]],A[rlist[l]])).mean(axis=0)
                try:
                    Cl_new = np.append(Cl_new, i)
                    A_new = np.vstack((A_new,A_tmp))
                except:
                    Cl_new = np.array([i])
                    A_new = A_tmp
        print('\n Number of original datapoints:', str(M[1:,1:].shape[0]))
        newMatrix = M
        temp = np.hstack((Cl_new[np.newaxis].T, A_new))
        newMatrix = np.vstack((newMatrix, temp))
        M = newMatrix
        print(' Number of new datapoints:', str(A_new.shape[0]))
        print(' Number of total datapoints:', str(newMatrix[1:,1:].shape[0]))

        del A_new, Cl_new, newMatrix
    return M

#************************************
''' Read Learning file '''
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
    A = M[1:,1:]
    Cl = M[1:,0]
    return M, En, A, Cl

#*****************************************
''' Write new learning '''
#*****************************************
def writeFile(File, M):
    if defParam.saveAsTxt == True:
        with open(File, 'ab') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    else:
        with h5py.File(File, 'w') as hf:
            hf.create_dataset("M",  data=M)

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
