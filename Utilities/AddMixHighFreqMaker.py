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
        print(' Usage:\n  python3 AddMixHighFreqMaker.py <learnData> <HighFreq threshold>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    M, En, A, Cl = readLearnFile(sys.argv[1])

    newTrainFile = os.path.splitext(sys.argv[1])[0] + '_av-hf' + sys.argv[2]

    if defParam.saveAsTxt == True:
        newTrainFile = newTrainFile + '.txt'
    else:
        newTrainFile = newTrainFile + '.h5'

    if os.path.exists(newTrainFile) == True:
        print(" Augmented training file with similar confuiguration exists. Exiting.\n")
        return

    A_new, Cl_new = selectHFdata(A, Cl, int(sys.argv[2]))
    writeFile(newTrainFile, M, En, A_new, Cl_new)
    print('\n Writing new training file: ', newTrainFile)
    
    print('\n Done!\n')
    
#************************************
''' Mix high frequency datasets '''
#************************************
def selectHFdata(A, Cl, HFthreshold):
    Cl_new = np.array([])
    #A_new = np.array([])
    
    uHFCl = np.array([])
    uniCl = np.unique(Cl).astype(int)
    for x in enumerate(uniCl):
        if np.count_nonzero(Cl==uniCl[x[0]]) >= HFthreshold:
            uHFCl = np.append(uHFCl, uniCl[x[0]])
    print(" Creating new datapoints by averaging members of the {0:.0f} classes with more than {1:.0f} datapoints...".format(uHFCl.shape[0], HFthreshold))
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
    
    print(" Created {0:.0f} new datapoints for each of the {1:.0f} classes with more than {2:.0f} elements.\n".format( Cl_new.shape[0],uHFCl.shape[0],HFthreshold))
    print(uHFCl)
          
    return A_new, Cl_new

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
def writeFile(File, M, En, A, Cl):
    print('\n Number of original datapoints:', str(M[1:,1:].shape[0]))
    print(' Number of new datapoints:', str(A.shape[0]))
    #newMatrix = np.append([0], En).reshape(1,-1)
    newMatrix = M
    temp = np.hstack((Cl[np.newaxis].T, A))
    newMatrix = np.vstack((newMatrix, temp))
    print(' Number of total datapoints:', str(newMatrix.shape[0]))

    if defParam.saveAsTxt == True:
        with open(File, 'ab') as f:
            np.savetxt(f, newMatrix, delimiter='\t', fmt='%10.6f')
    else:
        with h5py.File(File, 'w') as hf:
            hf.create_dataset("M",  data=newMatrix)

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
