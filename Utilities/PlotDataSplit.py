#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* Plot train data split in different files
* version: 20171013a
*
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)
import numpy as np
import sys, os.path, getopt, glob, csv
import matplotlib.pyplot as plt

def main():
    if(len(sys.argv)<2):
        print(' Usage:\n  python3 PlotDataSplit.py <learnData> <# spectra per plot>\n')
        print(' Usage (full set):\n  python3 PlotDataSplit.py <learnData>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    En, M, learnFileRoot = readLearnFile(sys.argv[1])
    try:
        nplots = int(sys.argv[2])
        if int(sys.argv[2])>=M.shape[0]:
            nplots = 1
    except:
        nplots = 1

    plotTrainData(En, M, learnFileRoot, nplots)

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
    #print ("En:",En)
    #print("M:",M)
    return En, M, learnFileRoot

#************************************
''' Plot data '''
#************************************
def plotTrainData(En, M, learnFileRoot, nplots):
    learnFileRootNew = learnFileRoot
    i=0
    j=0
    while j*nplots < M.shape[0]:
        if (j+1)*nplots < M.shape[0]:
            max = (j+1)*nplots
        else:
            max = M.shape[0]
        print(max)

        if nplots == 1:
            nplots = M.shape[0]
            max = M.shape[0]
            learnFileRootNew = learnFileRoot + '_full-set'
            plt.title(learnFileRoot+'\nFull set (#'+str(M.shape[0])+')')
        else:
            learnFileRootNew = learnFileRoot + '_partial-' + str(i)+'-'+str(max)
            plt.title(learnFileRoot+'\nPartial Set (#'+str(M.shape[0])+') ['+str(i)+', '+str(max)+']')

        print(' Plotting Training dataset in: ' + learnFileRootNew + '.png\n')
    
        while i < max:
            plt.plot(En, M[i,1:], label='Training data')
            i+=1
        plt.plot(En, M[0,1:], label='Training data')
        plt.xlabel('Raman shift [1/cm]')
        plt.ylabel('Raman Intensity [arb. units]')

        plt.savefig(learnFileRootNew + '.png', dpi = 160, format = 'png')  # Save plot
        #plt.show()
        plt.close()
        j+=1

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
