#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict - setup launcher
* Perform Machine Learning on Spectroscopy Data.
*
* Uses: Deep Neural Networks, TensorFlow, SVM, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
from .slp import *

#**********************************************
''' Main '''
#**********************************************
def main():
    run()
    
#************************************
''' Main initialization routine '''
#************************************
if __name__ == "__main__":
    sys.exit(main())
