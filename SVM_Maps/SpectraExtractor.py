#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

# Extract spectra of specific phases

import numpy as np
from sklearn import svm
from sklearn.externals import joblib

phaseColumn = 9
selPhase = 1
labelColumn = 2

mapfile = "Paraffin+cis-Azo2_map3_bs.txt"
clustfile = "Cluster_matrix-clust-all.txt"
newMapFile = "Paraffin_map3_phase1.txt"

f = open(mapfile, 'r')
A = np.loadtxt(f, unpack =False, skiprows=1)
A = np.delete(A, np.s_[0:2], 1)
f.close()
print(A.shape)

f = open(clustfile, 'r')
Cl = np.loadtxt(f, unpack =False, skiprows=1, usecols = range(phaseColumn-1,phaseColumn))
f.close()
print(Cl.shape)

f = open(clustfile, 'r')
L = np.loadtxt(f, unpack =False, skiprows=1, usecols = range(labelColumn-1,labelColumn))
f.close()
print(L.shape)

phaseMap = np.zeros(A.shape[1]+1)

for i in range(0,A.shape[0]):
    if Cl[i] == selPhase:
        temp = np.append(L[i], A[i,:])
        phaseMap = np.vstack((phaseMap, temp))

phaseMap = np.delete(phaseMap, np.s_[0:1], 0)

print(phaseMap.shape)
#print(phaseMap)

np.savetxt(newMapFile, phaseMap, delimiter=' ', fmt='%10.6f')

#f = open(newMapFile, 'r')
#B = np.loadtxt(f, unpack =False)
#f.close()

#print(B.shape)
#print(B)