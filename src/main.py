import cv2 as cv
import numpy as np
import images as im
from utlis import *
import tensorflow as tf
import os

#vid = cv.VideoCapture(0)

def mainloop():
        #tf.config.run_functions_eagerly(True)
        model = load_model(os.path.abspath('model.h5'))
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
        box = boxes[23]
        #box = cv.imread("img")
        pre = im.preprocess_box(box)
        pre = pre/255
        pre = pre[10:90, 10:90]
        pre = clean_box(pre)
        pre = cv.resize(pre,(28,28))
        cv.imshow("Box.png", pre)
        #convert to grayscale
        #gray = cv.cvtColor(box,cv.COLOR_RGB2GRAY)
        #ret1,threshold = cv.threshold(gray,127,255,cv.THRESH_BINARY_INV)
        box =  np.expand_dims(pre, axis=0)
        print(f"############{argmax(model.predict(box))}###############")

        cv.imshow("Contours", im.resize(cv.drawContours(conts, finded, 
                    -1, (0,255,0), 3), precent))
        cv.imshow("Approx", im.resize(cv.drawContours(approx_img, approx,
                    -1, (0,255,0), 20), precent))
        #cv.imshow("Predictions", im.resize(display_predictions(boxes,
        #            np.zeros((900,900,3))), precent))

        #cv.imwrite("box_cleaned.jpg", box)

        if len(approx) == 4:
            cv.imshow("Shrinked", im.resize(im.preprocess(im.cut_sudoku(shrinked,
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

