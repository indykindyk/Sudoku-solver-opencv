#!/usr/bin/env python
# coding: utf-8
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import (
    Activation,
    Convolution2D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling2D,
)
from tensorflow.keras.models import Sequential
import numpy as np
import pickle
import os

pickle_in = open(os.path.abspath("../data/X.pickle"),"rb")
X = pickle.load(pickle_in)

pickle_in = open(os.path.abspath("../data/y.pickle"),"rb")
y = pickle.load(pickle_in)

layers = [
    Convolution2D(128, 3, 3, input_shape=X.shape[1:], activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Convolution2D(64, 2, 1, activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Convolution2D(32, 2, 1, activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(1024, activation="relu"),
    Dropout(0.5),
    Dense(512, activation="relu"),
    Dropout(0.5),
    Dense(10),
    Activation("sigmoid"),
]

model = Sequential()
for layer in layers:
    model.add(layer)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        "/home/karol/python_projekty/cv2/sudoku/models/save_at_{epoch}.h5"
    ),
]
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
model.fit(np.array(X), np.array(y), batch_size=32, epochs=10)
