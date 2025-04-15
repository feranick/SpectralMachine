#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* ReadXmu
* Convert Xanes Xmu to ASCII
* File must be in xmu fomat
* version: v2025.04.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path

#************************************
# Main
#************************************
def main():
    if len(sys.argv) > 1:
        if os.path.isfile(sys.argv[1]):
            for ind, f in enumerate(sys.argv[1:]):
                saveFile(f,"")
        else:
            asciiPath = os.path.join(sys.argv[1], 'ASCII')
            if not os.path.isdir(asciiPath):
                os.mkdir(asciiPath)
            for ind, f in enumerate(os.listdir(sys.argv[1])):
                root=os.path.splitext(f)
                filePath = os.path.join(sys.argv[1], f)
                if os.path.isfile(filePath) and (root[-1] == ".txt") and (root[0][-5:] != "ASCII"):
                    saveFile(f,asciiPath)
    else:
        print(' Usage:\n')
        print(' (Single File): python3 ReadXmu.py <Xmu filename1> <Xmu filename2>, <Xmu filename3> ...\n')
        print(' (Batch conversion): python3 ReadXmu.py <folder>\n')
        print(' <folder> Directory where original Xmu files are located\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    print(' Done!\n')
    
#************************************
# Save File
#************************************
def saveFile(file, folder):
    try:
        with open(file, 'r') as f:
            #M = np.loadtxt(f, skiprows = 40, unpack=True)
            M = np.loadtxt(f, unpack=True)
        print(str(' ' + file) + '\n File OK, converting to ASCII... \n')
    
        newFile = os.path.join(folder, os.path.splitext(file)[0] + '_ASCII.txt')
    
        with open(newFile, 'w') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    except:
        print('\033[1m ' + str(sys.argv[1]) + ' file or folder not found \n' + '\033[0m')
        return

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
