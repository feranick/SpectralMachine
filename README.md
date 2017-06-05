# SpectralMachine
Machine learning software for rapid analysis of Raman spectra.

**Supported algorithms:**
 
 - Deep Neural Networks:
   - multi-layer perceptron (MLP) (L-BFGS Optimizer strongly recommended)
   - DNNClassifier (TensorFlow)
 - Support Vector Machine - SVM
 - TensorFlow (basic implementation)

**Additional multivariate analysis:**
- K-Means
- Principal component analysis

Installation
=============

This software requires Python (3.3 or higher). It has been tested with Python 3.5 or hiher which is the recommended platform. It is not compatible with python 2.x. Additional required packages:

    numpy
    scikit-learn (>=0.18)
    matplotlib 

These are found in Unix based systems using common repositories (apt-get for Debian/Ubuntu Linux, or MacPorts for MacOS). More details in the [scikit-learn installation page](http://scikit-learn.org/stable/install.html).

[TensorFlow](https://www.tensorflow.org) is needed only if flag is activated. Instructions for Linux and MacOS can be found in [TensorFlow installation page](https://www.tensorflow.org/install/). Pip installation is the easiest way to get going. Tested with TensorFlow v.1.0+ (not compatible with v.0.12 and below).


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


Data augmentation
==================
Using the provided training data set as is, accuracy is low. For a single training run using a random 30% of the training set for 100 times, the accuracy is about 32%:

./SpectraLearnPredict.py -t Training/20170227a/Training_kerogen_633nm_HC_20170227a.txt 1

Repeating the training few more times (5, for example) marginally increases the accuracy to 35.7% and it is fully converged. This is expected given the small dataset.

Increasing the number of spectra in the actual dataset can be done by accounting for noise. Using the AddNoisyData.py utility, the esisting training set is taken, and random variations in intensity at each energy point are added within a given offset. This virtually changes the overall spectra, without changing its overall trend in relation to the classifier (H:C). This allows for the preservation of the classifier for a given spectra, but it also increases the number of available spectra. This method is obviously a workaround, but it allows accuracy to be substantially increased. Furthermore, it lends a model better suited at dealing with noisy data. 

To recreate, let's start with adding noisy spectra to the training set. For example, let's add 5 replicated spectra with random noise added with an offset of 0.02 (intensity is normalized to the range [0,1])

AddNoisyData.py Training/20170227a/Training_kerogen_633nm_HC_20170227a.txt 5 0.02

Accuracy is increased to 80.4% with a single training run (100 times 30% of the dataset). 2 iterations increase accuracy to 95.8% and a third increased to 100%. (The same can be achieved by running 30% of the dataset 300 times).

One can optimize/minimize the number of spectra with added noise. Adding only 2 data-sets with noise offset at 0.02 converges the accuracy to about 94.6&. 
**One final word of caution**: Increasing the number of statistically independent available spectra for training is highly recommended over adding noisy data. 

More on Machine Learning tools used
====================================

- [Neural Networks](http://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [Deep Neural Networks - Multilayer Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Deep Neural Networks - TensorFlow](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier)
- [TensorFlow](https://www.tensorflow.org)
- [Support Vector Classification](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [K-Means](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Principal Component Analysis](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

