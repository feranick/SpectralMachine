#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
*******************************************************
* ConvMineralListCSVtoH5
* Convert Mineral List in CSV to H5
* version: v2024.11.14.1
* By: Nicola Ferralis <feranick@hotmail.com>
*******************************************************
'''
print(__doc__)

import pandas as pd
import sys, os.path

#************************************
# Main
#************************************
class dP:
    standardName = True

def main():
    try:
        File =  sys.argv[1]
        if os.path.splitext(File)[1] == ".csv" or os.path.splitext(File)[1] == ".CSV":
            print(" Opening CSV File:"+File)
            saveToHDF5(File)
    except:
        usage()
        sys.exit(2)

#************************************
# Convert CSV to HDF5
#************************************
def saveToHDF5(File):
    df = pd.read_csv(File, header=None).iloc[1:].drop([1,2,4], axis=1)
    print(df)
    
    if dP.standardName:
        newFile = "AAA_table_names.h5"
    else:
        fileRoot = os.path.splitext(File)[0]
        newFile = fileRoot+'.h5'
    
    df.to_hdf(newFile, key='df', mode='w')
    print("\n Original file: "+File)
    print(" New file: "+newFile+"\n")

#************************************
# Lists the program usage
#************************************
def usage():
    print(' Usage:\n  python3 ConvMineralListCSVtoH5.py.py <CSV File>.csv\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
