#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
* LabelFinder
* Find label corresponding to class in training data
* version: v2023.12.27.1
* By: Nicola Ferralis <feranick@hotmail.com>
*****************************************************
'''
print(__doc__)

import sys
#************************************
''' Main '''
#************************************
def main():
    if len(sys.argv) < 3:
        print(' Usage:\n  python3 LabelFinder <training_data_info_file.csv> <label1, label2, label3...>')
        print('  Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    import numpy as np
    learnFile = sys.argv[1]
    R = np.array([np.fromstring(sys.argv[2], dtype='uint64', sep=',')])
    
    print(" Opening learning info file:",learnFile,"\n")
    try:
        with open(learnFile, 'r') as f:
            M = f.readlines()
    except:
        print("\033[1m Learning Info File not found\033[0m")
        return

    for i in range(len(R[0])):
        findLabel(R[0,i], M)

#************************************
# Find label
#************************************
def findLabel(label, M):
    if int(label) > int(M[len(M)-1].split(',')[0]):
        print(" Check label, outside of range...\n")
        return

    for i in range(1,len(M)):
        if int(M[i].split(',')[0]) == label:
            print("  Label for class {0:s}: {1:s}\n".format(M[i].split(',')[0],
                        M[i].split(',')[3].split('__')[0]))
            break

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
