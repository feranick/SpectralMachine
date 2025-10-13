#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************
* Convert H5 into csv
* v2025.10.13.1
* Uses: TensorFlow, Keras
* By: Nicola Ferralis <feranick@hotmail.com>
**********************************************
'''
print(__doc__)
import sys, os.path, h5py
import pandas as pd

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3.py ConvertH5toCSV <file in HDF5 format>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        print(' Converting',sys.argv[1],'to CSV...')
        convertModelToTFJS(sys.argv[1])

#************************************
# Convert TF Model to TF.Lite
#************************************
def convertModelToTFJS(h5_file):
    csv_file = os.path.splitext(h5_file)[0]+".csv"
    df = pd.read_hdf(h5_file)
    df.to_csv(csv_file, index=False, header=False)
    print(f" Successfully converted {h5_file} to {csv_file}\n")
    
#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())


