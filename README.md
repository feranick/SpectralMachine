# SpectralMachine
Machine learning software for rapid analysis of Raman spectra.

Supported algorithms:
 - Support Vector Machine - SVM
 - Neural Network -  multi-layer perceptron (MLP)
 - TensorFlow

Additional multivariate analysis:
- Principal component analysis

Usage
======

- single files: 
  python SpectraLearnPredictSVM.py -f <learningfile> <spectrafile> 

- maps (formatted for Horiba LabSpec): 
  python SpectraLearnPredictSVM.py -m <learningfile> <spectramap> 

- batch txt files: 
  python SpectraLearnPredictSVM.py -b <learningfile> 


Training data
=============
Loosely based on Ferralis et al. [Carbon 108 (2016) 440](http://dx.doi.org/10.1016/j.carbon.2016.07.039).

Note
=====
This package requires [scikit-learn](http://scikit-learn.org/stable/), numpy and matplotlib.
[TensorFlow](https://github.com/tensorflow/tensorflow) is needed only if flag is activated.



