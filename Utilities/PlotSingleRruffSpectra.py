#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* PlotSingleRruffSpectra
* Plot Single Rruff spectra
* Files must be in RRuFF
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, getopt, glob, csv, re
from datetime import datetime, date
import matplotlib.pyplot as plt

def main():
    saveSummary = False
    savePlot = False
    saveAscii = False

    if len(sys.argv) < 2:
        print(' Usage:\n  python3 PlotSingleRruffSpectra.py <Spectra filename>\n')
        print(' Usage:\n  python3 PlotSingleRruffSpectra.py <Spectra filename> <EnIn> <EnFin> <EnStep>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        file =  sys.argv[1]

    rootPlotFile = "plot_"
    dateTimeStamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    plotFile = rootPlotFile+dateTimeStamp
    plt.figure(num=plotFile)

    with open(file, 'r') as f:
        En = np.loadtxt(f, unpack = True, usecols=range(0,1), delimiter = ',', skiprows = 10)
    with open(file, 'r') as f:
        R = np.loadtxt(f, unpack = True, usecols=range(1,2), delimiter = ',', skiprows = 10)

    if  len(sys.argv) < 4:
        enInit = float(En[0])
        enFin =  float(En[len(En)-1])
        enStep = float((enFin - enInit)/len(En))
    else:
        print("plotting with custom x-axes")
        enInit = sys.argv[2]
        enFin =  sys.argv[3]
        enStep = sys.argv[4]

    print(file + '\n File OK, converting to ASCII...')

    EnT = np.arange(float(enInit), float(enFin), float(enStep), dtype=np.float)
            
    if EnT.shape[0] == En.shape[0]:
        print(' Number of points in the learning dataset: ' + str(EnT.shape[0]))
    else:
        print('\033[1m' + ' Mismatch in datapoints: ' + str(EnT.shape[0]) + '; sample = ' +  str(En.shape[0]) + '\033[0m')
        # Interpolate to new axis
        R = np.interp(EnT, En, R, left = R[0], right = 0)
    # Renormalize offset by min R
    R = R - np.amin(R) + 1e-8
    # Renormalize to max of R
    R = R/np.amax(R)

    if saveAscii == True:
        try:
            convertFile = os.path.splitext(file)[0] + '_ASCII.txt'
            convertR = np.transpose(np.vstack((EnT, R)))
            with open(convertFile, 'ab') as f:
                np.savetxt(f, convertR, delimiter='\t', fmt='%10.6f')
        except:
            pass

    label = re.search('(.+?)__',file).group(1)
    if saveSummary == True:
        summaryPlotFile = rootPlotFile+"summary_"+dateTimeStamp+".csv"
        with open(summaryPlotFile, "a") as sum_file:
            sum_file.write('Classification started: '+dateTimeStamp+"\n")
            sum_file.write('0,,,' + label + ','+file+'\n')
    
    plt.plot(EnT,R,label=label)
    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')
    plt.legend(loc='upper left')
    if savePlot == True:
        plt.savefig(plotFile+".png", dpi = 160, format = 'png')  # Save plot
    plt.show()
    plt.close()

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
