# SpectralMachine: SpectraKeras
Machine learning software for rapid spectral analysis. While Raman spectra were the initilal focus, `SpectraKeras` is flexible to be applied for classification using any spectra (from XRD, FTIR and beyond). The previous generation software (`SpectraLearnPredict`) is no longer developed. 

**SpectraKeras**
- Currently supported ML architectures:
   - DNNClassifier (TensorFlow, TensorFlow-Lite)
   - Convolutional Neural Networks (TensorFlow, TensorFlow-Lite)
- Required libraries for prediction:
   - tensorflow (v2.16.2 recommended)
   - Optional: tensorflow-lite (v2.16.2 recommended)
   - Optional: [tensorflow-lite runtime](https://www.tensorflow.org/lite/guide/python) 
   - Optional: tensorflow-lite runtime with [Coral EdgeTPU](https://coral.ai/docs/accelerator/get-started/)

Credits and References
==================
If you use SpectralMachine or SpectraKeras, we request that you reference the papers/resources on which SpectralMachine is based:
- N Ferralis, SpectralMachine and SpectraKeras (2017), https://github.com/feranick/SpectralMachine
- N. Ferralis, R.E. Sumons, Automated Mineral and Geochemical Classification from Spectroscopy Data using Machine Learning, AGU Fall Meeting 2018, Washington, DC, December 2018, https://agu.confex.com/agu/fm18/prelim.cgi/Paper/369458 
- N. Ferralis, Automated Materials Classification from Spectroscopy/Diffraction through Deep Neural Networks, MRS Fall Meeting 2018, Boston, MA, November 2018, https://mrsfall2018.zerista.com/event/member/533306

Installation
=============
## Installation from available wheel package
If available from the main site, you can install SpectraKeras by running:

    python3 -m pip install --upgrade spectrakeras-2024.10.08.1-py3-none-any.whl
    
SpectraKeras_CNN and Spectrakeras_MLP are available directly from the command line.
NOTE: The Utilities in the `Utilities` folder are not included in the package, and can be run locally as needed.

## Make your own wheel package
Make sure you have the PyPA build package installed:

    python3 -m pip install --upgrade build
    
To build the wheel package rom the `SpectraKeras` folder run:

    python3 -m build
    
A wheel package is available in the subfolder `dir`. You can install it following the instructions shown above.

## Compatibility and dependences
This software requires Python (3.9 or higher). It has been tested with Python 3.9 or higher which is the recommended platform. It is not compatible with python 2.x. Additional required packages:

    numpy
    scikit-learn (>=0.18)
    scipy
    matplotlib
    pandas
    pydot
    graphviz
    h5py
    tensorflow (>=2.16.2)
    
In addition, these packages may be needed depending on your platform (via ```apt-get``` in debian/ubuntu or ```port``` in OSX):
    
    python3-tk
    graphviz

These are found in Unix based systems using common repositories (apt-get for Debian/Ubuntu Linux, or MacPorts for MacOS). More details in the [scikit-learn installation page](http://scikit-learn.org/stable/install.html).

[TensorFlow](https://www.tensorflow.org) is needed only if flag is activated. Instructions for Linux and MacOS can be found in [TensorFlow installation page](https://www.tensorflow.org/install/). Pip installation is the easiest way to get going. Tested with TensorFlow 2.x (2.15 or higher preferred). TF 2.15 is the currently supported release. 

Inference can be carried out using the regular tensorflow, or using [tensorflow-lite](https://www.tensorflow.org/lite/) for [quantized models](https://www.tensorflow.org/lite/performance/post_training_quantization). Loading times of tflite (direct or via [tflite-runtime](https://www.tensorflow.org/lite/guide/python)) are significantly faster than tensorflow with minimal loss in accuracy. SpectraKeras provides an option to convert tensorflow models to quantized tflite models. TFlite models have been tested in Linux x86-64, arm7 (including Raspberry Pi3) and aarm64, MacOS, Windows. 
    To use quantized models, TF 2.13 or higher is recommended. 

Inference using the [Coral EdgeTPU](https://coral.ai/) [tensorflow-lite](https://www.tensorflow.org/lite/) requires the [libedgetpu](https://github.com/google-coral/libedgetpu) libraries compatible with the supported and current version of `tflite-runtime` (version v2.13.0 or hiigher, v2.16.1 recommended), which is also required (Instructions and binaries can be found [here](https://github.com/feranick/TFlite-builds). More information on installation of such libraries at [Coral EdgeTPU](https://coral.ai/). 

Usage (SpectraKeras)
===================

## SpectraKeras_CNN

Train (Random cross validation):

    python3 SpectraKeras_CNN.py -t <learningFile>
    
Train (with external validation):

    python3 SpectraKeras_CNN.py -t <learningFile> <validationFile>
    
Predict:

    python3 SpectraKeras_CNN.py -p <testFile>
    
Batch predict:

    python3 SpectraKeras_CNN.py -b <folder>

Display Neural Netwrok Configuration:
    
    python3 SpectraKeras_CNN.py -n <learningFile>
    
Convert model to quantized tflite:
    
    python3 SpectraKeras_CNN.py -l <learningFile>
    
Determine accuracy using h5 testing file with spectra:

    python3 SpectraKeras_CNN.py -a <testFile>
    
## SpectraKeras_MLP

Train (Random cross validation):

    python3 SpectraKeras_MLP.py -t <learningFile>
        
Train (with external validation):

    python3 SpectraKeras_MLP.py -t <learningFile> <validationFile>
        
Predict:

    python3 SpectraKeras_MLP.py -p <testFile>
        
Batch predict:
        
    python3 SpectraKeras_MLP.py -b <folder>
        
Convert model to quantized tflite:
        
    python3 SpectraKeras_MLP.py -l <learningFile>
        
Determine accuracy using h5 testing file with spectra:

    python3 SpectraKeras_MLP.py -a <testFile>

Formatting input file for training
========================
The main idea behind the software is to train classification or regression models from plain spectra (which can be Raman, but they can be any spectra or diffraction profiles, as long as the model is consistent), rather than from manually selected features (such as bands, peaks, widths, etc). So, suppose one has training files similar to this, where the first column is the Raman shiift, the second is intensity:

```
1000  123
1001  140
1002  180
1003  150
...
```

Let's say this file correspond to label 1, and now one has a collection of files that will be used for training each with its own label, the input file will look like this:

```
0  1000  1001  1002  1003 ...
lab1  123 140  180  150  ...
lab2 ... ... ... ... ...
```
Essentially each line in the input file corresponds to a training file with its label. during training the model will learn (either through a simple deep MLP network using `SpectraKeras_MLP.py`, or through a Convolutional Network using `SpectraKeras_CNN.py`, which is recommended) to extract features needed for prediction. Note that all spectra needs to have the same Raman shifts max min and step.

Of course it is not expected that the user manually compiles the training file from a possibly large collection of spectra. For that, [`GenericDataMaker.py`](https://github.com/feranick/SpectralMachine/blob/master/Utilities/GenericDataMaker.py) is available in the [`Utilities`](https://github.com/feranick/SpectralMachine/tree/master/Utilities) folder, that can be used to automatically create such files. Basically you can run from the folder where you have your spectra:

    python3 GenericDataMaker.py <learnfile> <enInitial> <enFinal> <enStep>

The script will interpolate each spectra within the Raman shifts parameters you set above. Note that there are some basic configuration that you may need to change in the `GenericDataMakerp.py` for your case (such as delimiter between data, extension of the files, etc).

One can use the same to create a validation file, or you can use [other scripts](https://github.com/feranick/SpectralMachine/tree/master/Utilities) also provided to split the training set into training+validation. That can be done randomly within SpectraKeras, but the split will be different every time you run it.

Once models are trained trained, prediction on individual files can be made using simply formatted ASCII files (like in the example above).

Training data
=============
We do not provide advanced training sets, some of which can be found online. We only provide a simple Raman dataset mainly for testing purposes: it is loosely based on 633nm data from Ferralis et al. [Carbon 108 (2016) 440](http://dx.doi.org/10.1016/j.carbon.2016.07.039).

More on Machine Learning tools used
====================================

- [Neural Networks](http://scikit-learn.org/stable/modules/neural_networks_supervised.html)
- [Deep Neural Networks - Multilayer Perceptron](http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html)
- [Deep Neural Networks - TensorFlow](https://www.tensorflow.org/api_docs/python/tf/contrib/learn/DNNClassifier)
- [TensorFlow](https://www.tensorflow.org)
- [Support Vector Classification - deprecated](http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [K-Means - deprecated](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [Principal Component Analysis - deprecated](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

Known Issues
==================
None
