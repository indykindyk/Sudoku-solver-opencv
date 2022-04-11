import cv2 as cv
import numpy as np
import images as im
from utlis import *
import tensorflow as tf
import os
import sudoku

#vid = cv.VideoCapture(0)

def mainloop():
        while True: 
            model = load_model(os.path.abspath('model.h5'))
            #ret, cap = vid.read()
            #load sudoku image
            ret, cap = vid.read()
            print(cap)
            #cap = cv.imread(os.path.abspath("../images/new.jpg"))
            shrinked = cap.copy()
            img_preprocessed = im.preprocess(cap)
            conts = cap.copy()
            approx_img = cap.copy()
            precent = 25
            (h,w) = 4000,2248

            #find all contours form image
            finded = im.find_contours(img_preprocessed)
            #find corners from biggest contour
            approx = im.approx(finded)
            #split photo to 81 squares
            #shrinked_board = im.cut_sudoku(shrinked, approx)
            #boxes = split_photo(shrinked_board)
            #prediction_img, predictions, posarr = display_predictions(boxes, np.zeros((900,900,3)))
            #board = np.asarray(predictions)
            #board = board.reshape(9,9)
            #sudoku.solve(board)
            #print(board)
            #solved = np.reshape(board, (81))*posarr
            #print(solved)
            #solved_img, predictions, _ = display_predictions(solved, np.zeros((1152,1152,3)), solved=True)
            #simg = im.overlay(cap, solved_img, approx, w, h)
            cv.imshow("Contours", im.resize(cv.drawContours(conts, finded, 
                        -1, (0,255,0), 3), precent))
            cv.imshow("Approx", im.resize(cv.drawContours(approx_img, approx,
                        -1, (0,255,0), 20), precent))
            #cv.imshow("Predictions", im.resize(prediction_img,40))
            #cv.imshow("solved img", im.resize(simg,precent))        

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

            if len(approx) == 4:
                cv.imshow("Shrinked", im.resize(im.preprocess(im.cut_sudoku(shrinked,
                        approx)), precent))
            else:
                pass




# Read the original image
#cap = cv.VideoCapture(0)
#if not cap.isOpened():
#    print("cannot open camera")
#    exit()
#cap.release()o
if __name__ == '__main__':
    mainloop()
    cv.destroyAllWindows()

