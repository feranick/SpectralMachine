#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
* PlotRruffSpectraRound
* Plot Rruff spectra with rounding decimals
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
    if len(sys.argv) < 5:
        print(' Usage:\n  python3 PlotRruffSpectraRound.py <EnIn> <EnFin> <EnStep> <decimals>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        enInit = sys.argv[1]
        enFin =  sys.argv[2]
        enStep = sys.argv[3]
        decimals = int(sys.argv[4])

    rootPlotFile = "plot_"
    dateTimeStamp = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    summaryPlotFile = rootPlotFile+"summary_"+dateTimeStamp+".csv"
    plotFile = rootPlotFile+dateTimeStamp
    plt.figure(num=plotFile)

    with open(summaryPlotFile, "a") as sum_file:
                    sum_file.write('Classification started: '+dateTimeStamp+"\n")

    index = 0
    for ind, file in enumerate(sorted(os.listdir("."))):
        #try:
        if os.path.splitext(file)[-1] == ".txt":
            with open(file, 'r') as f:
                En = np.loadtxt(f, unpack = True, usecols=range(0,1), delimiter = ',', skiprows = 10)
            with open(file, 'r') as f:
                R = np.loadtxt(f, unpack = True, usecols=range(1,2), delimiter = ',', skiprows = 10)

            print(file + '\n File OK, converting to ASCII...')

            EnT = np.arange(float(enInit), float(enFin), float(enStep), dtype=float)
            
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
                
                R = np.around(R, decimals=decimals)
                
                index += 1
                '''
                try:
                    convertFile = os.path.splitext(file)[0] + '_ASCII.txt'
                    convertR = np.transpose(np.vstack((EnT, R)))
                    with open(convertFile, 'ab') as f:
                        np.savetxt(f, convertR, delimiter='\t', fmt='%10.6f')
                except:
                    pass
                '''
            label = re.search('(.+?)__',file).group(1)
            with open(summaryPlotFile, "a") as sum_file:
                sum_file.write(str(index) + ',,,' + label + ','+file+'\n')
    
            plt.plot(EnT,R,label=label)
        #except:
        #    print("\n Skipping: ",file)

    plt.xlabel('Raman shift [1/cm]')
    plt.ylabel('Raman Intensity [arb. units]')
    plt.legend(loc='upper right')
    plt.savefig(plotFile+".png", dpi = 160, format = 'png')  # Save plot
    plt.show()
    plt.close()

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
