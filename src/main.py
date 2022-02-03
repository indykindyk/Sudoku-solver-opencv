import cv2 as cv
import numpy as np
import images as im
from utlis import *
import tensorflow as tf

#vid = cv.VideoCapture(0)

def mainloop():
        model = load_model('/home/karol/python_projekty/cv2/sudoku/models/save_at_12.h5')
        #ret, cap = vid.read()
        cap = cv.imread("/home/karol/python_projekty/cv2/sudoku/images/new.jpg")
        shrinked = cap.copy()
        img_preprocessed = im.preprocess_box(cap)
        conts = cap.copy()
        approx_img = cap.copy()
        precent = 25

        #find all contours form image
        finded = im.find_contours(img_preprocessed)
        #find corners from biggest contour
        approx = im.approx(finded)
        #split photo to 81 squares
        boxes = split_photo(im.cut_sudoku(shrinked, approx))

        cv.imshow("Box", boxes[2])

        box = tf.expand_dims(boxes[2], 0)
        print(box.shape)
        print(f"############{argmax(model.predict(box))}###############")

        cv.imshow("Contours", im.resize(cv.drawContours(conts, finded, 
                    -1, (0,255,0), 3), precent))
        cv.imshow("Approx", im.resize(cv.drawContours(approx_img, approx,
                    -1, (0,255,0), 20), precent))
        cv.imshow("Predictions", im.resize(display_predictions(boxes,
                    np.zeros((900,900,3))), precent))

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

