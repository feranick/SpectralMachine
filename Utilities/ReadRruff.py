#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*********************************************
*
* ReadRruFF
* Convert RRuFFspectra to ASCII
* File must be in RRuFF
* version: 20191025a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, getopt, glob, csv

#************************************
# Main
#************************************
def main():
    
    if len(sys.argv) < 2:
        print(' Usage:\n')
        print(' (Single File): python3 ReadRruff.py <folder> <RRuFF filename>\n')
        print(' (Batch conversion): python3 ReadRruff.py <folder>\n')
        print(' <folder> Directory where ASCII files are to be saved\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    if len(sys.argv) < 3:
        for ind, f in enumerate(sorted(os.listdir("."))):
            if (os.path.splitext(f)[-1] == ".txt"):
                saveFile(sys.argv[1],f)
    else:
        saveFile(sys.argv[1],sys.argv[2])

    print(' Done!\n')
    
#************************************
# Save File
#************************************
def saveFile(folder, file):
    print(file)
    try:
        with open(file, 'r') as f:
            M = np.loadtxt(f, skiprows = 10, delimiter = ',', unpack=False)
        print(str(' ' + file) + '\n File OK, converting to ASCII... \n')
        newFile = folder+os.path.splitext(file)[0] + '_ASCII.txt'
        with open(newFile, 'ab') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    except:
        print('\033[1m ' + str(sys.argv[1]) + ' file not found \n' + '\033[0m')
        return

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
