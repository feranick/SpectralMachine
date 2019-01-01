#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
**********************************************************
* Convert TF models (TF, Keras) into TF.Lite
* 20181231a
* Uses: TensorFlow
* By: Nicola Ferralis <feranick@hotmail.com>
***********************************************************
'''
print(__doc__)
import tensorflow as tf
import sys, os.path, h5py

#************************************
# Main
#************************************
def main():
    if len(sys.argv) < 2:
        print(' Usage:\n  python3.py <Learning File in HDF5 format>\n')
        print(' Requires python 3.x. Not compatible with python 2.x\n')
        return
    else:
        print(' Converting',sys.argv[1],'to TF.Lite...\n')
        convertModelToTFLite(sys.argv[1])

#************************************
# Convert TF Model to TF.Lite
#************************************
def convertModelToTFLite(model):
    modelRoot = os.path.splitext(model)[0]
    try:
        converter = tf.lite.TFLiteConverter.from_keras_model_file(model)
    except:
        converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file(model)
    tflite_model = converter.convert()
    open(modelRoot+".tflite", "wb").write(tflite_model)
    print(' Converted model saved to:',modelRoot+".tflite\n")

#************************************
# Main initialization routine
#************************************
if __name__ == "__main__":
    sys.exit(main())


