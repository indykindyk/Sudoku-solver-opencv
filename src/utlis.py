import numpy as np
from numpy import argmax, imag, mean, amax
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2 as cv
import images as im
from scipy import ndimage as ndi


model = load_model('models/model-04-1.00.h5')	

predictions = []

def split_photo(img):
    #split photo into 81 squares
    vsplit = np.vsplit(img, 9)
    boxes = []
    for vs in vsplit:
        hsplit = np.hsplit(vs, 9)
        for hs in hsplit:
            boxes.append(hs)
    return boxes

# load an image and predict the class
def predict(boxes):
    # load model
    x = 0
    #give prediction for evry square
    predictions = []
    for img in boxes:
        pre = im.preprocess_box(img)
        name = f"box{x}.png"
        pre = im.prepare_box(pre)
        
        x+=1

        if pre.sum() >= 28**2*255 - 28 * 1 * 255:
            predictions.append(0)
            continue    # Move on if we have a white cell

        center_width = pre.shape[1] // 2
        center_height = pre.shape[0] // 2
        x_start = center_height // 2
        x_end = center_height // 2 + center_height
        y_start = center_width // 2
        y_end = center_width // 2 + center_width
        center_region = pre[x_start:x_end, y_start:y_end]

        box =  np.expand_dims(pre, axis=0)
        box = box/255.
        predict = model.predict(box)
        digit = argmax(predict)
        digit += 1
        print(f"[{x}] pred: {digit}")
        predictions.append(digit)

    return np.array(predictions)

def display_predictions(boxes, solved=False):
    predictions = []
    global posArray
    if not solved:
        predictions = predict(boxes)
        posArray = np.where(predictions > 0,0,1)
    else:
        predictions = []
        predictions = boxes
        posArray = None

    img = np.zeros((1152,1152,3))
    imgW = img.shape[1]/9
    imgH = img.shape[0]/9
    for x in range(9):
        for y in range(9):
            if predictions[(y*9)+x] != 0:
                cv.putText(img, str(predictions[(y*9)+x]),
                (int(x*imgW+imgW/2-10), int((y+0.7)*imgH)),
                cv.FONT_HERSHEY_SIMPLEX, 2.5, (255,0,255), 2,
                cv.LINE_AA)

    return img, predictions, posArray

