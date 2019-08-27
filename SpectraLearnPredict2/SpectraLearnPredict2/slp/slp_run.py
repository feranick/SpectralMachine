#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict2
* Perform Machine Learning on Spectroscopy Data.
*
* Uses: Deep Neural Networks, TensorFlow, SVM, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''

import matplotlib
if matplotlib.get_backend() == 'TkAgg':
    matplotlib.use('Agg')

import numpy as np
import sys, os.path, getopt, glob, csv
import random, time, configparser, os
from os.path import exists, splitext
from os import rename
from datetime import datetime, date

from .slp_config import *
from .slp_io import *

#**********************************************
''' Main '''
#**********************************************
def run():

    start_time = time.clock()
    print(" TensorFlow v.",parse_version(tf.version.VERSION) )

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
            try:
                trainAccuracy(sys.argv[2], testFile)
            except:
                usage()
                sys.exit(2)

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
        print(" TensorFlow v.",parse_version(tf.version.VERSION) )
        print(" Total time: {0:.1f}s or {1:.1f}m or {2:.1f}h".format(total_time,
                                                    total_time/60, total_time/3600),"\n")

#************************************
''' Lists the program usage '''
#************************************
def usage():
    print('\n Usage:\n')
    print(' Single files:')
    print('  python3 SpectraLearnPredict2.py -f <learningfile> <spectrafile>\n')
    print(' Cross-validation for accuracy determination:')
    print('  python3 SpectraLearnPredict2.py -a <learningfile> <testdataset>\n')
    print(' Cross-validation for accuracy determination (automatic splitting):')
    print('  python3 SpectraLearnPredict2.py -a <learningfile>\n')
    print(' Maps (formatted for Horiba LabSpec):')
    print('  python3 SpectraLearnPredict2.py -m <learningfile> <spectramap>\n')
    print(' Batch txt files:')
    print('  python3 SpectraLearnPredict2.py -b <learningfile>\n')
    print(' K-means on maps:')
    print('  python3 SpectraLearnPredict2.py -k <spectramap> <number_of_classes>\n')
    print(' Principal component analysis on spectral collection files: ')
    print('  python3 SpectraLearnPredict2.py -p <spectrafile> <#comp>\n')
    print(' Requires python 3.x. Not compatible with python 2.x\n')
