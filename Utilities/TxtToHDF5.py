#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
***********************************************************
* TxtToHDF5
* Convert txt-formatted learning data into HDF5
* version: v2023.12.15.1
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)

import numpy as np
import h5py, sys, os.path, getopt

#************************************
''' Main '''
#************************************
class defParam:
    Ynorm = False
    YnormTo = 1
    saveNormAsTxt = False

def main():
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "nkh:", ["norm", "key", "help"])
        if len(sys.argv) < 2:
            usage()
            return
        else:
            if "-n" in [x[0] for x in opts] :
                defParam.Ynorm = True
            file =  args[0]
        saveLearnFile(file)
    except:
        usage()
        sys.exit(2)

#************************************
''' Convert Learning file to HDF5 '''
#************************************
def saveLearnFile(learnFile):
    learnFileRoot = os.path.splitext(learnFile)[0]
    learnFileNorm = learnFileRoot+'_norm'
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Learning file not found \n' + '\033[0m')
        return
    
    #En = M[0,1:]
    #A = M[1:,1:]
    #Cl = M[1:,0]

    if defParam.Ynorm:
        print(" Normalizing spectra to:",defParam.YnormTo)
        A = M[1:,1:]
        for i in range(0,A.shape[0]):
            if(np.amin(A[i]) <= 0):
                A[i,:] = A[i,:] - np.amin(A[i,:]) + 1e-8
            A[i,:] = np.multiply(A[i,:], defParam.YnormTo/max(A[i][:]))
        M[1:,1:] = A

        if defParam.saveNormAsTxt == True:
            if os.path.isfile(learnFileNorm+'.txt') is False:
                with open(learnFileNorm+'.txt', 'ab') as f:
                    np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
                print(" Normalized training spectra saved in:",learnFileNorm+'.txt',"\n")
            else:
                print(" Normalized training file in txt format already exists. Not saving...\n")
        learnFileRoot = learnFileNorm

    if os.path.isfile(learnFileRoot+'.h5') is False:
        with h5py.File(learnFileRoot+'.h5', 'w') as hf:
            #hf.create_dataset("En",  data=En)
            hf.create_dataset("M",  data=M)
            #hf.create_dataset("Cl",  data=Cl.astype('|S9'))
            #hf.create_dataset("A",  data=A)
        print(" Learning file converted to hdf5: "+learnFileRoot+".h5\n")

    else:
            print(" Normalized training file in hdf5 format already exists. Not saving... \n")

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print(' Usage:\n  python3 TxtToHDF5.py <Learning File DataMaker>\n')
    print(' With Normalization:\n  python3 TxtToHDF5.py -n <Learning File prepared with DataMaker>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
