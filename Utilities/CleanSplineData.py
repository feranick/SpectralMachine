#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************
* CleanSplineData
* Remove repeated data, spline
* File must be in ASCII
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************
'''
print(__doc__)

import numpy as np
import sys, os.path, getopt, glob, csv
from scipy import interpolate
import matplotlib.pyplot as plt

#************************************
# Main
#************************************
class dP:
    plotInData = False
    plotAvData = False
    plotSplData = False
    splineData = False
    xStep = 0.5
    
    if splineData:
        folderTag = "_spline/"
        fileTag = "_spline.txt"
    else:
        folderTag = "_clean/"
        fileTag = "_clean.txt"
    
def main():
    
    if len(sys.argv) < 2:
        print(' Usage:\n')
        print(' (Single File): python3 SplineData.py <folder> <ASCII filename>\n')
        print(' (Batch conversion): python3 SplineData.py <folder>\n')
        print(' <folder> Directory where ASCII files are to be saved\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    
    Params = dP()
    
    if len(sys.argv) < 3:
        try:
            os.mkdir(sys.argv[1] + Params.folderTag)
        except:
            pass
        for ind, f in enumerate(os.listdir(sys.argv[1])):
            root=os.path.splitext(f)
            if (root[-1] == ".txt") and (root[0][-5:] != "spline") and (root[0][-5:] != "clean"):
                saveFile(sys.argv[1],f,1)
    else:
        saveFile(sys.argv[1],sys.argv[2],0)

    print(' Done!\n')
    
#************************************
# Save File
#************************************
def saveFile(folder, file, type):
    try:
        Params = dP()
        with open(folder+'/'+file, 'r') as f:
            M = np.loadtxt(f, skiprows = 0, delimiter = '\t', unpack=False)
        print(str(' ' + file) + '\n File OK, continuing... ')

        if dP.plotInData:
            plot(M[:,0],M[:,1])
    
        ### Rescaled in y from 0.
        print(' Rescaling data...')
        M=np.vstack([M[:,0],M[:,1]-np.min(M[:,1])]).T
    
        ### Find repeated xvalues and average the corresponding yvalues
        print(' Remove and replace repeated data...')
        vals, indices = np_unique_indices(M[:,0])
        exclIndex = np.array([], dtype=np.int8)
        averRows= []
        for val, idx in zip(vals, indices):
            if len(idx[0]) > 1:
                exclIndex = np.append(exclIndex,idx[0])
                averRows.append(np.mean(M[idx][0],axis=0))

        ### Purge repeated xvalues, add averaged values for previously repeated values
        M1=np.delete(M,exclIndex, axis=0)
        if len(averRows)>0:
            M1=np.vstack([M1,np.array(averRows)])
            M1 = M1[M1[:,0].argsort()]
        if dP.plotAvData:
            plot(M1[:,0],M1[:,1])
    
        ###spline data
        if dP.splineData:
            print(' Spline data...')
            spl = interpolate.splrep(M1[:,0], M1[:,1])
            xspl = np.arange(min(spl[0]),max(spl[0]),dP.xStep)
            yspl = interpolate.splev(xspl, spl)
            M1 = np.hstack([xspl.reshape(-1,1),yspl.reshape(-1,1)])
    
            if dP.plotSplData:
                plot(xspl, yspl)

        if type == 1:
            newFile = folder + Params.folderTag + os.path.splitext(file)[0] + Params.fileTag
        else:
            newFile = folder + '/' + os.path.splitext(file)[0] + fileTag
        with open(newFile, 'ab') as f:
            np.savetxt(f, M1, delimiter='\t', fmt='%10.6f')
        print(' Data saved in:',newFile,'\n')
    
    except:
        print('\033[1m ' + str(sys.argv[1]) + ' file not found \n' + '\033[0m')
        return
    
#************************************
# Find unique indexes
#************************************
def np_unique_indices(arr, **kwargs):
    """Unique indices for N-D arrays."""
    vals, indices, *others = np_unique_indices_1d(arr.reshape(-1), **kwargs)
    indices = [np.stack(np.unravel_index(x, arr.shape)) for x in indices]
    return vals, indices, *others

def np_unique_indices_1d(arr, **kwargs):
    """Unique indices for 1D arrays."""
    sort_indices = np.argsort(arr)
    arr = np.asarray(arr)[sort_indices]
    vals, first_indices, *others = np.unique(
        arr, return_index=True, **kwargs
    )
    indices = np.split(sort_indices, first_indices[1:])
    for x in indices:
        x.sort()
    return vals, indices, *others
    
def plot(x,y):
    plt.plot(x,y,'ro')
    plt.plot(x,y, 'b')
    plt.show()
    

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())
