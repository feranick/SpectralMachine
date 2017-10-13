#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Change the Xrange
* version: 20171013a
*
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)
import numpy as np
import sys, os.path, random
import matplotlib.pyplot as plt

def main():
    if(len(sys.argv)<2):
        print(' Usage:\n  python3 XRange.py <learnData> <min> <max>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, M, P, learnFileRoot = readLearnFile(sys.argv[1])
    try:
        min = int(sys.argv[2])
        max = int(sys.argv[3])
        En, M = resize(En, M, min, max)
    except:
        pass
    
    plotTrainData(En, M, learnFileRoot)
    saveCvs(P,En,M,learnFileRoot)

#************************************
''' Open Learning Data '''
#************************************
def readLearnFile(learnFile):
    try:
        with open(learnFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Learn data file not found \n' + '\033[0m')
        return

    learnFileRoot = os.path.splitext(learnFile)[0]

    P = M[1:,0]
    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(np.array(M[:,1:]),np.s_[0:1],0)

    print("En:",En.shape)
    print("M:",M.shape)
    print("Param:", P.shape)
    #print ("En:",En)
    #print("M:",M)
    return En, M, P, learnFileRoot

def resize(En, M, min, max):
    En = En[min:(max)]
    M = M[:,min:max]
    print(En.shape)
    print(M.shape)
    return En, M

def saveCvs(P,En,M,learnFileRoot):
    
    trainFile = learnFileRoot+"_test.txt"
    newTrain = np.append([0], En)
    for i in range(0,M.shape[0]):

        newTrain = np.vstack((newTrain, np.append(float(P[i]),M[i,:])))

    with open(trainFile, 'ab') as f:
        np.savetxt(f, newTrain, delimiter='\t', fmt='%10.6f')

#************************************
''' Plot data '''
#************************************
def plotTrainData(En, M, learnFileRoot):
    learnFileRootNew = learnFileRoot

    start = 0
    learnFileRootNew = learnFileRoot + '_full-set'
    plt.title(learnFileRoot+'\nFull set (#'+str(M.shape[0])+')')

    print(' Plotting Training dataset in: ' + learnFileRootNew + '.png\n')
    
    for i in range(start,M.shape[0], 1):
        plt.plot(En, M[i,:], label='Training data')
    
    plt.plot(En, M[0,:], label='Training data')

    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')

    plt.savefig(learnFileRootNew + '.png', dpi = 160, format = 'png')  # Save plot
    
    plt.show()
    plt.close()


#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
