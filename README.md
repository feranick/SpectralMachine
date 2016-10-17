# SpectralMachine
Machine learning software for rapid analysis of Raman spectra.

Supported algorithms:
 - Support Vector Machine - SVM
 - Neural Network -  multi-layer perceptron (MLP)
 - TensorFlow

Additional multivariate analysis:
- Principal component analysis

Installation
=============

This software requires Python (2.6 or higher, 3.3 or higher). It has been tested with Python 3.5 which is the recommended platform.

This package requires [scikit-learn](http://scikit-learn.org/stable/), numpy and matplotlib. These are found in Unix based systems using common repositories (apt-get for Debian/Ubuntu Linux, or MacPorts for MacOS). More details in the [scikit-learn installation page](http://scikit-learn.org/stable/install.html).

[TensorFlow](https://github.com/tensorflow/tensorflow) is needed only if flag is activated. Instructions for Linux and MacOS can be found in [TensorFlow installation page](https://www.tensorflow.org/versions/r0.11/get_started/os_setup.html). Pip installation is the easiest way to get going.


Usage
======

- Single files: 
  
    python SpectraLearnPredictSVM.py -f learningfile spectrafile 

- Maps (formatted for Horiba LabSpec): 
  
    python SpectraLearnPredictSVM.py -m learningfile spectramap 

- Batch txt files:

    python SpectraLearnPredictSVM.py -b learningfile 


Training data
=============
Loosely based on Ferralis et al. [Carbon 108 (2016) 440](http://dx.doi.org/10.1016/j.carbon.2016.07.039).


More on Machine Learning tools used
====================================

- [Support Vector Classification](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Neural Networks](http://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [TensorFlow](https://www.tensorflow.org/)
- [Principal Component Analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

