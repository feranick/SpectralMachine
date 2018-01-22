#!/usr/bin/env python3
# https://goo.gl/AH2dsU

import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation

cat = cv2.imread('cat.png')
print(cat.shape)
fig = plt.figure()
plt.imshow(cat)
plt.show()

model = Sequential()
model.add(Conv2D(3,(3,3), input_shape=cat.shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

cat_batch = np.expand_dims(cat, axis=0)
conv_cat = model.predict(cat_batch)

def visualize_cat(cat_batch):
    cat = np.squeeze(cat_batch, axis=0).astype(np.uint8)
    print(cat)
    print(cat.shape)
    fig = plt.figure()
    plt.imshow(cat)
    plt.show()

visualize_cat(conv_cat)


