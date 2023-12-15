#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************
* Convert TF models (TF, Keras) into TF.js
* v2023.12.15.1
* Uses: TensorFlow, Keras
* By: Nicola Ferralis <feranick@hotmail.com>
**********************************************
'''
print(__doc__)
import tensorflow as tf
import tensorflowjs as tfjs
import sys, os.path, h5py
import keras
#import tensorflow.keras as keras

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3.py ConvertToTFJS <model in HDF5 format>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        print('\n Converting',sys.argv[1],'to TF.js...\n')
        convertModelToTFLite(sys.argv[1])

#************************************
# Convert TF Model to TF.Lite
#************************************
def convertModelToTFLite(savedModel):
    model = keras.models.load_model(savedModel)
    convFile = os.path.splitext(savedModel)[0]+'_js'
    tfjs.converters.save_keras_model(model, convFile)
    print(' Converted model saved inside:',convFile,'\n')

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())


