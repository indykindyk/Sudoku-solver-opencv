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
import cv2 as cv
import numpy as np
import os
from tqdm import tqdm
import random

CATEGORIES = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
train_ds = []

IMG_SIZE = 128
batch_size = 50


def create_training_data():
    DATADIR = "/home/karol/python_projekty/cv2/sudoku/data/train"
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            img_array = cv.imread(os.path.join(path, img))
            gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
            ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
            train_ds.append([thresh, class_num])


create_training_data()
print(len(train_ds))

random.shuffle(train_ds)

X = []
y = []

for features, label in train_ds:
    X.append(features)
    y.append(label)

print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y).reshape(-1, 1)

X = X / 255.0

X.shape

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
