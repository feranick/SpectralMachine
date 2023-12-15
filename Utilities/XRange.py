#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* Change the Xrange
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)
import numpy as np
import sys, os.path, random, h5py
import matplotlib.pyplot as plt

#************************************
''' Main '''
#************************************
class defParam:
    saveAsTxt = False

def main():
    if(len(sys.argv)<2):
        print(' Usage:\n  python3 XRange.py <learnData> <min x-axis> <max x-axis>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, Cl, M, learnFileRoot = readLearnFile(sys.argv[1])
    try:
        min = int(sys.argv[2])
        max = int(sys.argv[3])
        En, M = resize(En, M, min, max)
        learnFileRoot = learnFileRoot+"_range-"+str(min)+"-"+str(max)
        saveNewLearnFile(En,Cl,M,learnFileRoot)
        plotTrainData(En, M, learnFileRoot)
    except:
        print("\n Training file not changed. Exit\n")

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
    
    learnFileRoot = os.path.splitext(learnFile)[0]
    En = M[0,1:]
    A = M[1:,1:]
    Cl = M[1:,0]
    print("  Classes:", Cl.shape)
    print("  En:",En.shape)
    print("  M:",M.shape)
    return En, Cl, A, learnFileRoot

#************************************
''' Resize Learning Data '''
#************************************
def resize(En, M, min, max):
    indMin = np.where(En>min)[0][0]
    indMax = np.where(En>max)[0][0]
    En = En[indMin:indMax]
    M = M[:,indMin:indMax]
    print("  New En:", En.shape)
    print("  New M:", M.shape)
    return En, M

#************************************
''' Save New Learning Data '''
#************************************
def saveNewLearnFile(En,Cl,M,learnFile):
    newTrain = np.append([0], En)
    
    for i in range(0,M.shape[0]):
        newTrain = np.vstack((newTrain, np.append(float(Cl[i]),M[i,:])))

    if defParam.saveAsTxt == True:
        learnFile += '.txt'
        print(" Saving new training file (txt) in:", learnFile+"\n")
        with open(learnFile, 'ab') as f:
            np.savetxt(f, newTrain, delimiter='\t', fmt='%10.6f')
    else:
        learnFile += '.h5'
        print(" Saving new training file (hdf5) in: "+learnFile+"\n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=newTrain)

#************************************
''' Plot data '''
#************************************
def plotTrainData(En, M, learnFileRoot):
    learnFileRootNew = learnFileRoot

    start = 0
    learnFileRootNew = learnFileRoot + '_full-set'
    plt.title(learnFileRoot+'\nRestricted X Range')

    print('  Plotting Training dataset in: ' + learnFileRootNew + '.png\n')
    
    for i in range(start,M.shape[0], 1):
        plt.plot(En, M[i,:], label='Training data')
    
    plt.plot(En, M[0,:], label='Training data')

    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')
    plt.savefig(learnFileRootNew + '.png', dpi = 160, format = 'png')  # Save plot
    #plt.show()
    plt.close()

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
