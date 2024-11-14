#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*******************************************************
* GetSpectraNames
* Get names of minerals from prediction codes
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
*******************************************************
'''
print(__doc__)

import pandas as pd
import sys, os.path

#************************************
# Main
#************************************
def main():
    try:
        File =  sys.argv[1]
        if os.path.splitext(File)[1] == ".h5":
            print(" Opening h5 File:"+File)
            getMineral(File, sys.argv[2])
    except:
        usage()
        sys.exit(2)

#************************************
# Get name of prediction
#************************************
def getMineral(File, pred):
    df = pd.read_hdf(File)
    name = df[df[0]==str(pred)].iloc[0,1]
    ind = name.find('__')
    print(ind)
    print(name[:ind])
    
#************************************
# Lists the program usage
#************************************
def usage():
    print(' Usage:\n  python3 GetSpectraNames.py <CSV File> <pred value>.csv\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
