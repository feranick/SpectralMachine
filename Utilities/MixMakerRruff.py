#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* MixMakerRruff
* Mix different rruff files into a ASCII
* Files must be in RRuFF
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, glob, csv, re, h5py
from datetime import datetime, date
import matplotlib.pyplot as plt

#************************************
''' Main '''
#************************************
class defParam:
    saveAsTxt = True
    saveAsASCII = False
    plotData = False

def main():
    if len(sys.argv) < 4:
        print(' Usage:\n  python3 MixMakerRruff.py <EnIn> <EnFin> <EnStep> (<threshold %>)\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        enInit = sys.argv[1]
        enFin =  sys.argv[2]
        enStep = sys.argv[3]
        if len(sys.argv)<5:
            print("No threshold defined, setting to zero")
            threshold = 0
        else:
            threshold = sys.argv[4]
            print("Setting threshold to:", threshold,"%")

    rootMixFile = "mixture"
    dateTimeStamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    mixFile = rootMixFile+"_"+dateTimeStamp
    summaryMixFile = rootMixFile+"-summary_"+dateTimeStamp+".csv"
    plotFile = rootMixFile+"-plot_"+dateTimeStamp
    plt.figure(num=plotFile)

    with open(summaryMixFile, "a") as sum_file:
                    sum_file.write('Classification started: '+dateTimeStamp+\
                    ",enInit="+str(enInit)+",enFin="+str(enFin)+",enStep="+str(enStep)+"\n")
    index = 0
    first = True

    for ind, file in enumerate(sorted(os.listdir("."))):
        try:
            if file[:7] != "mixture" and os.path.splitext(file)[-1] == ".txt" and file[-10:] != "_ASCII.txt":
                with open(file, 'r') as f:
                    En = np.loadtxt(f, unpack = True, usecols=range(0,1), delimiter = ',', skiprows = 10)
                with open(file, 'r') as f:
                    R = np.loadtxt(f, unpack = True, usecols=range(1,2), delimiter = ',', skiprows = 10)
                    
                R[R<float(threshold)*np.amax(R)/100] = 0
                print('\n' + file + '\n File OK, converting to ASCII...')

                EnT = np.arange(float(enInit), float(enFin), float(enStep), dtype=float)
            
                if EnT.shape[0] == En.shape[0]:
                    print(' Number of points in the learning dataset: ' + str(EnT.shape[0]))
                else:
                    print('\033[1m' + ' Mismatch in datapoints: ' + str(EnT.shape[0]) + '; sample = ' +  str(En.shape[0]) + '\033[0m')

                # Interpolate to new axis
                R = np.interp(EnT, En, R, left = R[0], right = 0)
                # Renormalize offset by min R
                R = R - np.amin(R)
                # Renormalize to max of R
                R = R/np.amax(R)
                    
                if first:
                    mixR = R
                    first = False
                else:
                    mixR = (mixR*index + R)/(index+1)
                index += 1

                print('\033[1m' + ' Mismatch corrected: datapoints in sample: ' + str(R.shape[0]) + '\033[0m')

                if defParam.saveAsASCII == True:
                    saveAsASCII(EnT,R,file)
                
                label = re.search('(.+?)__',file).group(1)
                with open(summaryMixFile, "a") as sum_file:
                    sum_file.write(str(index) + ',,,' + label + ','+file+'\n')
    
                if defParam.plotData == True:
                    plt.plot(EnT,R,label=label)
        except:
            print("\n Skipping: ",file)

    try:
        newR = np.transpose(np.vstack((EnT, mixR)))
    except:
        print("No usable files found\n ")
        return

    saveMixFile(newR, mixFile)

    if defParam.plotData == True:
        plt.plot(EnT, mixR, linewidth=3, label=r'Mixture')
        plt.xlabel('Raman shift [1/cm]')
        plt.ylabel('Raman Intensity [arb. units]')
        plt.legend(loc='upper right')
        plt.savefig(plotFile+".png", dpi = 160, format = 'png')  # Save plot
        plt.show()
        plt.close()

#***************************************
''' Save new learning Data '''
#***************************************
def saveMixFile(M, learnFile):
    if defParam.saveAsTxt == True:
        learnFile += '.txt'
        print("\n Saving mixture file (txt) in:\033[1m", learnFile+"\033[0m \n")
        with open(learnFile, 'ab') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    else:
        learnFile += '.h5'
        print("\n Saving mixture file (hdf5) in:\033[1m"+learnFile+"\033[0m \n")
        with h5py.File(learnFile, 'w') as hf:
            hf.create_dataset("M",  data=M)

#***************************************
''' Save mixture file in ASCII '''
#***************************************
def saveAsASCII(EnT, R, file):
    try:
        convertFile = os.path.splitext(file)[0] + '_ASCII.txt'
        convertR = np.transpose(np.vstack((EnT, R)))
        with open(convertFile, 'ab') as f:
            np.savetxt(f, convertR, delimiter='\t', fmt='%10.6f')
        print(" Saving spectra as ASCII in: "+file)
    except:
        print(" Saving spectra as ASCII failed")

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
