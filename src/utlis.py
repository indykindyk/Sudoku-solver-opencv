import numbers
import numpy as np
from numpy import argmax, imag, mean, amax
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2 as cv
import images as im
from scipy import ndimage

model = load_model('model.h5')	

predictions = []

def split_photo(img):
	#split photo into 81 squares
    try:
        if len(img) > 1:
            vsplit = np.vsplit(img, 9)
            boxes = []
            for vs in vsplit:
                hsplit = np.hsplit(vs, 9)
                for hs in hsplit:
                    boxes.append(hs)
            return boxes
    except TypeError:
        pass

def clean_box(img):
	w, h = img.shape
	cy,cx = ndimage.measurements.center_of_mass(img)
	shiftx = np.round(w/2.0-cx).astype(int)
	shifty = np.round(h/2.0-cy).astype(int)
	M = np.float32([[1,0,shiftx],[0,1,shifty]])
	shifted = cv.warpAffine(img,M,(w,h))
	return shifted

def load_image(img):
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img

# load an image and predict the class
def predict(boxes):
	# load model
	x = 1
	#give prediction for evry square
	for img in boxes:
		# load the imagev
		#img = load_image(img)
		# predict the class
		pre = im.preprocess_box(img)
		pre = pre/255
		pre = pre[10:90, 10:90]
		pre = clean_box(pre)
		pre = cv.resize(pre,(28,28))

		if pre.sum() <= 28**2*255 - 28 * 1 * 255:
			predictions.append(0)
			continue
		
		box =  np.expand_dims(pre, axis=0)
		predict = model.predict(box)
		digit = argmax(predict)
		#get the probability value
		probability_value = amax(predict)
		print(f"[{x}] pred: {digit}, conf: {round(probability_value)} %")
		x+=1
		predictions.append(digit+1)

	return np.asarray(predictions)

def display_predictions(boxes, img, solved=False):
	global posArray
	if not solved:
		predictions = predict(boxes)
		posArray = np.where(predictions > 0,0,1)
	else:
		predictions = boxes
		posArray = None

	imgW = img.shape[1]/9
	imgH = img.shape[0]/9
	for x in range(9):
		for y in range(9):
			if predictions[(y*9)+x] != 0:
				cv.putText(img, str(predictions[(y*9)+x]),
				(int(x*imgW+imgW/2-10), int((y+0.8)*imgH)),
				cv.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 2,
				cv.LINE_AA)

	return img, predictions, posArray

