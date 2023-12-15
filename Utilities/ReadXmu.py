#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* ReadXmu
* Convert Xanes Xmu to ASCII
* File must be in xmu format
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)


import numpy as np
import sys, os.path, getopt, glob, csv

def main():
    if(len(sys.argv)<2):
        print(' Usage:\n  python3 readxmu.py <Xmu filename>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return

    try:
        with open(sys.argv[1], 'r') as f:
            #M = np.loadtxt(f, skiprows = 40, unpack=True)
            M = np.loadtxt(f, unpack=True)
        print(str(' ' + sys.argv[1]) + '\n File OK, converting to ASCII... \n')
    except:
        print('\033[1m ' + str(sys.argv[1]) + ' file not found \n' + '\033[0m')
        return

    print(M[0])

    newFile = os.path.splitext(sys.argv[1])[0] + '_b.txt'

    with open(newFile, 'ab') as f:
        np.savetxt(f, M, delimiter='\t', fmt='%10.6f')

    print(' Done!\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
