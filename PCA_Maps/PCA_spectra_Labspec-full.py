#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
#*********************************************
#
# PCA_spectra_Labspec-full
# Perform SVM machine learning on Raman maps.
# version: 20160916a
#
# By: Nicola Ferralis <feranick@hotmail.com>
#
#**********************************************
'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.decomposition import PCA

numPCAcomp = 10

mapfile = "map.txt"
clustfile = "clust.txt"
kernel = 'rbf'  #Use either 'linear' or 'rbf' (for large number of features)

sampleFile = "Sample.txt"
whichColumn = 2

print(" Opening files and parameters...")

f = open(mapfile, 'r')
A = np.loadtxt(f, unpack =False, skiprows=1)
A = np.delete(A, np.s_[0:2], 1)
f.close()
print(A)
print(A.shape)

En = np.c_[0:A.shape[1]]
print(En)

f = open(clustfile, 'r')
Cl = np.loadtxt(f, unpack =False, skiprows=1, usecols = range(whichColumn-1,whichColumn))
f.close()
#Cl = [round(x*1000) for x in Cl]
Cl = ['{:.2f}'.format(x) for x in Cl]
#print(Cl)

print(' Running PCA...\n')
pca = PCA(n_components=numPCAcomp)
pca.fit(A)
        
print(pca.explained_variance_ratio_)
print(pca.components_.shape)

for i in range(0,pca.components_.shape[0]):
    plt.plot(En, pca.components_[i,:], label='PCA')
plt.show()
