#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
*
* SpectraLearnPredict
* Perform Machine Learning on Spectroscopy Data.
* version: 20171221a
*
* Uses: Deep Neural Networks, TensorFlow, SVM, PCA, K-Means
*
* By: Nicola Ferralis <feranick@hotmail.com>
*
***********************************************************
'''
print(__doc__)

from .slp_run import *
from .slp_config import *
from .slp_io import *
from .slp_preprocess import *
from .slp_tf import *
from .slp_nn import *
from .slp_svm import *
from .slp_pca import *
from .slp_kmeans import *

