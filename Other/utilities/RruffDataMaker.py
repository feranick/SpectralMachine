#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
*********************************************
*
* RRuffDataMaker
* Adds spectra to single file for classification
* File must be in RRuFF
* version: 20170306a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, glob
from datetime import datetime, date

#**********************************************
''' main '''
#**********************************************

def main():
    try:
        processMultiFile(sys.argv[1])
    except:
        usage()
    sys.exit(2)

#**********************************************
''' Open and process inividual files '''
#**********************************************
def processMultiFile(learnFile):
    index = 0
    learnFileRoot = os.path.splitext(learnFile)[0]
    summary_filename = learnFileRoot + str(datetime.now().strftime('_%Y-%m-%d_%H-%M-%S.log'))
    with open(summary_filename, "a") as sum_file:
        sum_file.write(str(datetime.now().strftime('Classification started: %Y-%m-%d %H:%M:%S\n')))
    for f in glob.glob('*.txt'):
        if (f != learnFile):
            makeFile(f, learnFile, index)
            index = index + 1
            with open(summary_filename, "a") as sum_file:
                sum_file.write(str(index) + '\t' + f +'\n')

#**********************************************
''' Add data to Training file '''
#**********************************************
def makeFile(sampleFile, learnFile, param):
    try:
        with open(sampleFile, 'r') as f:
            En = np.loadtxt(f, unpack = True, usecols=range(0,1), delimiter = ',', skiprows = 10)
        with open(sampleFile, 'r') as f:
            R = np.loadtxt(f, unpack = True, usecols=range(1,2), delimiter = ',', skiprows = 10)
            print(' Number of points in \"' + sampleFile + '\": ' + str(En.shape[0]))
    except:
        print('\033[1m' + ' Sample data file not found \n' + '\033[0m')
        return

    EnT = np.arange(100, 1500, 1, dtype=np.float)
    if EnT.shape[0] == En.shape[0]:
        print(' Number of points in the learning dataset: ' + str(EnT.shape[0]))
    else:
        print('\033[1m' + ' Mismatch in datapoints: ' + str(EnT.shape[0]) + '; sample = ' +  str(En.shape[0]) + '\033[0m')
        R = np.interp(EnT, En, R)
        print('\033[1m' + ' Mismatch corrected: datapoints in sample: ' + str(R.shape[0]) + '\033[0m')


    if os.path.exists(learnFile):
        newTrain = np.append(float(param),R).reshape(1,-1)
    else:
        print('\n\033[1m' + ' Train data file not found. Creating...' + '\033[0m')
        newTrain = np.append([0], EnT)
        print(' Added spectra to \"' + learnFile + '\"\n')
        newTrain = np.vstack((newTrain, np.append(float(param),R)))

    with open(learnFile, 'ab') as f:
        np.savetxt(f, newTrain, delimiter='\t', fmt='%10.6f')

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:')
    print('  python ClassDataMaker.py <learnfile>\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
