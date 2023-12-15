#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************
* RemoveLimitedDatasets
* Remove data with little representation based on threshold
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

def main():
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 RemoveLimitedDatasets.py <learnData> <threshold>')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    newFile = os.path.splitext(sys.argv[1])[0] + '_above' + sys.argv[2]
    exclFile = os.path.splitext(sys.argv[1])[0] + '_excludeBelow' + sys.argv[2]
    learnFileExt = os.path.splitext(sys.argv[1])[1]

    En, M = readLearnFile(sys.argv[1])

    numClasses = np.unique(M[:,0]).size
    indClass = np.zeros((numClasses))
    rosterSpectra = np.zeros((M.shape[0]))
    totNumIncl = 0

    le = MultiClassReductor()
    le.fit(np.unique(M[:,0]))
    Cl = le.transform(M[:,0])

    # sort how many spectra per class
    for i in range(M.shape[0]):
        indClass[int(Cl[i])]+=1

    # create roster for spectra above threshold
    for i in range(M.shape[0]):
        if indClass[int(Cl[i])] >= float(sys.argv[2]):
            rosterSpectra[i] = 1
            totNumIncl += 1

    # find out how many classes are above the threshold
    for i in range(numClasses):
        if indClass[i] >= float(sys.argv[2]):
            print(" Class: ",i, "- spectra per class:", int(indClass[i]))
        else:
            print(" Class: ",i, "- spectra per class:", int(indClass[i])," - EXCLUDED")

    totClassIncl = indClass[np.where(indClass < float(sys.argv[2]))]

    print("\n Number of points per spectra:", M[0,1:].size)
    print("\n Original number of unique classes:", numClasses)
    print(" Number of included unique classes:",totClassIncl.shape[0])
    print(" Number of excluded unique classes:",indClass.shape[0] - totClassIncl.shape[0])

    print("\n Original number of spectra in training set:", M.shape[0])
    print(" Number of spectra included in new training set:", totNumIncl)
    print(" Number of spectra excluded in new training set:", M.shape[0] - totNumIncl,"\n")

    # create new training set above threshold
    print(" Creating new training dataset with included spectra...")
    newTrain = M[np.where(rosterSpectra == 1.)[0],:]
    newTrain = np.insert(newTrain,0,np.append([0], En),0)

    # create new training set below threshold
    print(" Creating new training dataset with the excluded spectra... \n")
    exclTrain = M[np.where(rosterSpectra == 0.)[0],:]
    exclTrain = np.insert(exclTrain,0,np.append([0], En),0)
    
    saveLearnFile(newTrain, newFile)
    saveLearnFile(exclTrain, exclFile)

#************************************
''' Open Learning Data '''
#************************************
def readLearnFile(learnFile):
    print(" Opening training file: "+learnFile+"\n")
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
# MultiClassReductor
#************************************
class MultiClassReductor():
    def __self__(self):
        self.name = name
    
    def fit(self,tc):
        self.totalClass = tc.tolist()
    
    def transform(self,y):
        Cl = np.zeros(y.shape[0])
        for j in range(len(y)):
            Cl[j] = self.totalClass.index(np.array(y[j]).tolist())
        return Cl
    
    def inverse_transform(self,a):
        return [self.totalClass[int(a)]]

    def classes_(self):
        return self.totalClass

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
