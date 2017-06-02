# SpectralMachine
Machine learning software for rapid analysis of Raman spectra.

Supported algorithms:
 
 - Deep Neural Networks:
   - multi-layer perceptron (MLP) (L-BFGS Optimizer strongly recommended)
   - DNNClassifier (TensorFlow)
 - Support Vector Machine - SVM
 - TensorFlow (basic implementation)

Additional multivariate analysis:
- K-Means
- Principal component analysis

Installation
=============

This software requires Python (3.3 or higher). It has been tested with Python 3.5 or hiher which is the recommended platform. It is not compatible with python 2.x.

This package requires:

    numpy
    scikit-learn (>=0.18)
    matplotlib 

These are found in Unix based systems using common repositories (apt-get for Debian/Ubuntu Linux, or MacPorts for MacOS). More details in the [scikit-learn installation page](http://scikit-learn.org/stable/install.html).

[TensorFlow](https://www.tensorflow.org) is needed only if flag is activated. Instructions for Linux and MacOS can be found in [TensorFlow installation page](https://www.tensorflow.org/install/). Pip installation is the easiest way to get going. Tested with TensorFlow v.1.0 (not compatible with v.0.12 and below).


Usage
======

Single files: 
  
    python3 SpectraLearnPredict.py -f learningfile spectrafile 

Maps (formatted for Horiba LabSpec): 
  
    python3 SpectraLearnPredict.py -m learningfile spectramap 

Batch txt files:

    python3 SpectraLearnPredict.py -b learningfile 

K-means on Raman maps:
    
    python3 SpectraLearnPredict.py -k spectramap number_of_classes

Principal component analysis on spectral collection files:
    
    python3 SpectraLearnPredict.py -p spectrafile #comp


Training data
=============
Loosely based on 633nm data from Ferralis et al. [Carbon 108 (2016) 440](http://dx.doi.org/10.1016/j.carbon.2016.07.039).


More on Machine Learning tools used
====================================

- [Neural Networks](http://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [Deep Neural Networks - Multilayer Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Deep Neural Networks - TensorFlow](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier)
- [TensorFlow](https://www.tensorflow.org)
- [Support Vector Classification](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [K-Means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Principal Component Analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

