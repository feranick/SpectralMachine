#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*****************************************************
*
* TextBinConverter
* Convert training files to/from text from/to bin
* version: 20180301a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
*******************************************************
'''
print(__doc__)

import numpy as np
import sys, os.path

#**********************************************
''' main '''
#**********************************************
def main():
    
    if len(sys.argv) < 2:
        usage()
        return
        
    try:
        if os.path.splitext(sys.argv[1])[1] == ".txt":
            text2bin(sys.argv[1])
        else:
            bin2text(sys.argv[1])
    except:
        usage()

    sys.exit(2)

#**********************************************
''' Text to Binary format '''
#**********************************************
def text2bin(learnFile):
    learnFileRoot = os.path.splitext(learnFile)[0]
    with open(learnFile, 'r') as f:
        M = np.loadtxt(f, unpack =False)
    print(M)
    print(M.shape)
    file = open(learnFileRoot+".npy", 'ab')
    np.save(file, M, '%10.6f')
    print("Training file saved as binary in:", learnFileRoot+".npy")

#**********************************************
''' Binary to Text format '''
#**********************************************
def bin2text(learnFile):
    learnFileRoot = os.path.splitext(learnFile)[0]
    M = np.load(learnFile)
    
    print(M)
    print(M.shape)
    with open(learnFileRoot+".txt", 'ab') as f:
        np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
    print("Training file saved as text in:", learnFileRoot+".txt")

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print('  python3 TextBinConverter.py <learnfile> \n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
