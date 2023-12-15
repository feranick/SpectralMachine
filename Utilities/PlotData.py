#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Plot train data skipping with steps
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)
import numpy as np
import sys, os.path, random, h5py
import matplotlib.pyplot as plt

def main():
    if(len(sys.argv)<2):
        print(' Usage:\n  python3 PlotData.py <learnData> <step>\n')
        print(' Usage (full set):\n  python3 PlotData.py <learnData>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    try:
        step = int(sys.argv[2])
    except:
        step = 1
    En, M, learnFileRoot = readLearnFile(sys.argv[1])
    plotTrainData(En, M, learnFileRoot, step)

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
    #Cl = M[1:,0]
    return En, A, learnFileRoot

#************************************
''' Plot data '''
#************************************
def plotTrainData(En, M, learnFileRoot, step):
    learnFileRootNew = learnFileRoot
    if step == 1:
        start = 0
        learnFileRootNew = learnFileRoot + '_full-set'
        plt.title(learnFileRoot+'\nFull set (#'+str(M.shape[0])+')')
    else:
        start = random.randint(0,10)
        learnFileRootNew = learnFileRoot + '_partial-' + str(step) + '_start-' + str(start)
        plt.title(learnFileRootNew+'\nPartial Set (#'+str(M.shape[0])+'): every '+str(step)+' spectrum, start at: '+ str(start))

    print(' Plotting Training dataset in: ' + learnFileRootNew + '.png\n')
    
    for i in range(start,M.shape[0], step):
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
