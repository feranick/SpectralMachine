#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************************
* RangeToDataMaker
* version: v2023.12.27.1
* run: python3 RangeTiDataMaer <csv file from Google doc>
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
#print(__doc__)

import sys, math, os.path, time, pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

#************************************
''' Main '''
#************************************
def main():
    ##########################
    # Parameters
    ##########################

    fullSet = True
    numCols = 8

    colToChange1 = 2
    rangeSteps = 10
    
    minRange1 = 1
    maxRange1 = 10
    dataRange1 = []
    for i in range(rangeSteps):
        dataRange1.append(minRange1 + i*(maxRange1-minRange1)/(rangeSteps-1))
    
    file = sys.argv[1]
    
    #************************************
    ''' Read File '''
    #************************************
    df = pd.read_csv(file)
    if fullSet == True:
        numCols = len(df.columns)
    A = np.array(df.values[:,:numCols],dtype=np.float64)
    print(A.shape)
    print(A)

    for i in range(rangeSteps):
        A[0,colToChange1]=dataRange1[i]
        print(A)
        newFile = os.path.splitext(file)[0] + "_" + str(i) + ".csv"
        #np.savetxt(newFile, A, fmt='%10.1f', delimiter=',')
        A.tofile(newFile, sep=',', format='%10.1f')


#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
