import numpy as np
import os
from tqdm import tqdm
import cv2 as cv
import pickle
import random
import images as im
#from utlis import clean_box


class dataset:
    def __init__(self):
        self.CATEGORIES = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        self.train_ds = []

        self.IMG_SIZE = 28
        self.batch_size = 50

        self.X = []
        self.y = []
        self.DATADIR = "/home/karol/model_training_data/train"

    def create_training_data(self):
        for category in self.CATEGORIES:
            path = os.path.join(self.DATADIR, category)
            class_num = self.CATEGORIES.index(category)
            for img in tqdm(os.listdir(path)):
                img_array = cv.imread(os.path.join(path, img))
                gray = cv.cvtColor(img_array, cv.COLOR_RGB2GRAY)
                ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY)
                #pre = clean_box(thresh)
                self.train_ds.append([cv.resize(thresh, (28, 28)), class_num])

    def save(self):
        random.shuffle(self.train_ds)

        for features, label in self.train_ds:
            self.X.append(features)
            self.y.append(label)

        X = np.array(self.X).reshape(-1, self.IMG_SIZE, self.IMG_SIZE, 1)
        y = np.array(self.y).reshape(-1, 1)

        pickle_out = open("/home/karol/model_training_data/X.pickle", "wb")
        pickle.dump(X, pickle_out)
        pickle_out.close()

        pickle_out = open("/home/karol/model_training_data/y.pickle", "wb")
        pickle.dump(y, pickle_out)
        pickle_out.close()

data = dataset()
data.create_training_data()
data.save()
