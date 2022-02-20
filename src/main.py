import cv2 as cv
import numpy as np
import images as im
from utlis import *
import tensorflow as tf
import os

#vid = cv.VideoCapture(0)

def mainloop():
        model = load_model(os.path.abspath('../models/save_at_10.h5'))
        #ret, cap = vid.read()
        #load sudoku image
        cap = cv.imread(os.path.abspath("../images/new.jpg"))
        shrinked = cap.copy()
        img_preprocessed = im.preprocess(cap)
        conts = cap.copy()
        approx_img = cap.copy()
        precent = 25

        #find all contours form image
        finded = im.find_contours(img_preprocessed)
        #find corners from biggest contour
        approx = im.approx(finded)
        #split photo to 81 squares
        boxes = split_photo(im.cut_sudoku(shrinked, approx))

        box = cv.imread("img009-00039.png")
        #box = im.preprocess_box(box)
        cv.imshow("Box", box)
        #convert to grayscale
        gray = cv.cvtColor(box,cv.COLOR_RGB2GRAY)
        ret1,threshold = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV)
        box = threshold.reshape(-1,128,128,1)
        tf.expand_dims(box, 0)
        print(f"############{argmax(model.predict(box))}###############")

        cv.imshow("Contours", im.resize(cv.drawContours(conts, finded, 
                    -1, (0,255,0), 3), precent))
        cv.imshow("Approx", im.resize(cv.drawContours(approx_img, approx,
                    -1, (0,255,0), 20), precent))
        #cv.imshow("Predictions", im.resize(display_predictions(boxes,
        #            np.zeros((900,900,3))), precent))

        #cv.imwrite("box_cleaned.jpg", box)

        if len(approx) == 4:
            cv.imshow("Shrinked", im.resize(im.preprocgrayscaleess(im.cut_sudoku(shrinked,
                       approx)), precent))
        else:
            pass

        cv.waitKey(0)



# Read the original image
#cap = cv.VideoCapture(0)
#if not cap.isOpened():
#    print("cannot open camera")
#    exit()
#cap.release()o
if __name__ == '__main__':
    mainloop()
    cv.destroyAllWindows()

