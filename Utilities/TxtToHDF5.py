#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* TxtToHDF5
* Convert txt-formatted learning data into HDF5
*
* version: 20180611a
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
print(__doc__)

import numpy as np
import h5py, sys, os.path

#************************************
''' Main '''
#************************************
class defParam:
    Ynorm = False
    YnormTo = 1

def main():

    if len(sys.argv) < 2:
        print(' Usage:\n  python3 TxtToHDF5.py <Learning File prepared with RruffDataMaker>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        file =  sys.argv[1]

    saveLearnFile(file)

#************************************
''' Convert Learning file to HDF5 '''
#************************************
def saveLearnFile(learnFile):
    learnFileRoot = os.path.splitext(learnFile)[0]
    try:
        if os.path.splitext(learnFile)[1] == ".npy":
            M = np.load(learnFile)
        else:
            with open(learnFile, 'r') as f:
                M = np.loadtxt(f, unpack =False)
    except:
        print('\033[1m' + ' Learning file not found \n' + '\033[0m')
        return

    #En = np.delete(np.array(M[0,:]),np.s_[0:1],0)
    #M = np.delete(M,np.s_[0:1],0)
    #Cl = np.asarray(['{:.2f}'.format(x) for x in M[:,0]])
    #A = np.delete(M,np.s_[0:1],1)
    
    #En = M[0,1:]
    #A = M[1:,1:]
    #Cl = M[1:,0]

    if defParam.Ynorm ==True:
        learnFileNorm = learnFileRoot+'_norm'
        print(" Normalizing spectra to:",defParam.YnormTo)
        A = M[1:,1:]
        YnormXind = np.where(M[0,1:]>0)[0].tolist()
        for i in range(0,A.shape[0]):
            A[i,:] = np.multiply(A[i,:], defParam.YnormTo/A[i,A[i][YnormXind].tolist().index(max(A[i][YnormXind].tolist()))+YnormXind[0]])
        M[1:,1:] = A

        with open(learnFileNorm+'.txt', 'ab') as f:
            np.savetxt(f, M, delimiter='\t', fmt='%10.6f')
        print(" Normalized training spectra saved in:",learnFileNorm+'.txt',"\n")
        learnFileRoot = learnFileNorm

    with h5py.File(learnFileRoot+'.h5', 'w') as hf:
        #hf.create_dataset("En",  data=En)
        hf.create_dataset("M",  data=M)
        #hf.create_dataset("Cl",  data=Cl.astype('|S9'))
        #hf.create_dataset("A",  data=A)

    print(" Learning file converted to hdf5: "+learnFileRoot+".h5\n")

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
