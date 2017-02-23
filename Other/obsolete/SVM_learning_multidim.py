#!/usr/bin/python3.4
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn import svm


data = np.array([[1,2,3,5,6],[3,4,5,6,7],[6,7,8,9,10],[9,10,11,12,13]])
label = np.array([1,0,1,0])

print(data)
print(label)

print(data.shape)
print(label.shape)

print(round(31.123))

clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(data,label)

#f = open('5a.txt', 'r')
#R = np.loadtxt(f, unpack =True, usecols=range(1,2))
#R = R.reshape(1,-1)
#f.close()

#print(clf.predict(R))