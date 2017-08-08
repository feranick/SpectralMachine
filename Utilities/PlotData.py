#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* Plot data
*
* version: 20170807a
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
        print(' Usage:\n  python3 PlotData.py <learnData>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, M = readLearnFile(sys.argv[1])
    
    plotTrainData(En, M)

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

    En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    M = np.delete(M,np.s_[0:1],0)
    
    print("En:",En.shape)
    print("M:",M.shape)
    print ("En:",En)
    print("M:",M)
    return En, M

#************************************
''' Plot data '''
#************************************
def plotTrainData(En, M):
    import matplotlib.pyplot as plt

    #print(' Plotting Training dataset in: ' + learnFileRoot + '.png\n')
    for i in range(0,M.shape[0], 1):
        plt.plot(En, M[i,1:], label='Training data')
    
    plt.plot(En, M[0,1:], label='Training data')

    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')

    #plt.savefig(learnFileRoot + '.png', dpi = 160, format = 'png')  # Save plot
    
    plt.show()
    plt.close()

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
