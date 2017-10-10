#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict - standalone launcher
* Perform Machine Learning on Spectroscopy Data.
* Version 20171007e
*
* Uses: Deep Neural Networks, TensorFlow, SVM, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
print(__doc__)

import matplotlib
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Agg')

import numpy as np
import sys, os.path, getopt, glob, csv
import random, time, configparser, os
from os.path import exists, splitext
from os import rename
from datetime import datetime, date

from slp import *

#**********************************************
''' Main '''
#**********************************************
def main():

    start_time = time.clock()
    try:
        opts, args = getopt.getopt(sys.argv[1:],
                                   "fambkph:", ["file", "accuracy", "map", "batch",
                                   "kmaps", "pca", "help"])
    except:
        usage()
        sys.exit(2)

    if opts == []:
        usage()
        sys.exit(2)

    print(" Using training file: ", sys.argv[2],"\n")
    for o, a in opts:
        if o in ("-f" , "--file"):
            try:
                LearnPredictFile(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)

        if o in ("-a" , "--accuracy"):
            print('\033[1m Running in cross validation mode for accuracy determination...\033[0m\n')
            try:
                if sys.argv[3]:
                    testFile = sys.argv[3]
            except:
                preprocDef.subsetCrossValid = True
                testFile = "tmp"
            #try:
            trainAccuracy(sys.argv[2], testFile)
            #except:
            #    usage()
            #    sys.exit(2)

        if o in ("-m" , "--map"):
            try:
                LearnPredictMap(sys.argv[2], sys.argv[3])
            except:
                usage()
                sys.exit(2)

        if o in ("-b" , "--batch"):
            try:
                LearnPredictBatch(sys.argv[2])
            except:
                usage()
                sys.exit(2)

        if o in ("-p" , "--pca"):
            if len(sys.argv) > 3:
                numPCAcomp = int(sys.argv[3])
            else:
                numPCAcomp = pcaDef.numPCAcomponents
            try:
                runPCA(sys.argv[2], numPCAcomp)
            except:
                usage()
                sys.exit(2)

        if o in ("-k" , "--kmaps"):
            if len(sys.argv) > 3:
                numKMcomp = int(sys.argv[3])
            else:
                numKMcomp = kmDef.numKMcomponents
            try:
                KmMap(sys.argv[2], numKMcomp)
            except:
                usage()
                sys.exit(2)
        total_time = time.clock() - start_time
        print(" Total time (s):",total_time)

#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
