#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* Plot train data
*
* version: 20171012a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)


import numpy as np
import sys, os.path, getopt, glob, csv

def main():
    if(len(sys.argv)<2):
        print(' Usage:\n  python3 PlotData.py <learnData> <step>\n')
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
    try:
        with open(learnFile, 'r') as f:
            M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Learn data file not found \n' + '\033[0m')
        return

    learnFileRoot = os.path.splitext(learnFile)[0]

    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(M,np.s_[0:1],0)
    
    print("En:",En.shape)
    print("M:",M.shape)
    print ("En:",En)
    print("M:",M)
    return En, M, learnFileRoot

#************************************
''' Plot data '''
#************************************
def plotTrainData(En, M, learnFileRoot, step):
    import matplotlib.pyplot as plt

    #print(' Plotting Training dataset in: ' + learnFileRoot + '.png\n')
    if step == 1:
        learnFileRoot = learnFileRoot + '_full-set'
    else:
        learnFileRoot = learnFileRoot + '_partial-' + str(step)

    print(' Plotting Training dataset in: ' + learnFileRoot + '.png\n')
    
    for i in range(0,M.shape[0], step):
        plt.plot(En, M[i,1:], label='Training data')
    
    plt.plot(En, M[0,1:], label='Training data')

    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')

    plt.savefig(learnFileRoot + '.png', dpi = 160, format = 'png')  # Save plot
    
    plt.show()
    plt.close()

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
