#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*******************************************************
* CSVToTxt
* Convert CSV-formatted learning data into ASCII Txt
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
*******************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path

#************************************
''' Main '''
#************************************
class dP:
    skipRows = 2

def main():
    try:
        if len(sys.argv)<2:
            for ind, f in enumerate(sorted(os.listdir("."))):
                if os.path.splitext(f)[1] == ".csv" or os.path.splitext(f)[1] == ".CSV":
                    print(" Opening CSV File:"+f)
                    saveFile(f)
        else:
            File =  sys.argv[1]
            if os.path.splitext(File)[1] == ".csv" or os.path.splitext(File)[1] == ".CSV":
                print(" Opening CSV File:"+File)
                saveFile(File)
    except:
        usage()
        sys.exit(2)

#************************************
''' Convert Learning file to HDF5 '''
#************************************
def saveFile(File):

    fileRoot = os.path.splitext(File)[0]
    try:
        M = np.loadtxt(open(File, "rb"), delimiter=",", skiprows=dP.skipRows)
    except:
        print('\033[1m' + ' Input file not found \n' + '\033[0m')
        return

    if os.path.isfile(fileRoot+'.txt') is False:
        with open(fileRoot+'.txt', 'wb') as f:
            np.savetxt(f, M, delimiter='    ')
        print(" File: "+File+" converted into ASCII\n")
    else:
            print(" File in txt format already exists. Not saving... \n")

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print(' Usage:\n  python3 CSVToTxt.py <CSV File>.csv\n')
    print(' Run an all csv in a folder:\n  python3 CSVToTxt.py\n')    
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
