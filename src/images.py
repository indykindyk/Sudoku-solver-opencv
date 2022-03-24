import cv2 as cv
import numpy as np
import utlis

def preprocess(img):
    """
    This function preprocess the image.
    It performs basic operations like
    applaying a blur and treshold etc.
    """
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    #add blur
    gray = cv.GaussianBlur(gray, (9, 9), 0)
    #add bibteralFilter to reduce noise
    frame = cv.bilateralFilter(gray,9,75,75)
    #applay inverted treshold
    threshold = cv.adaptiveThreshold(frame,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv.THRESH_BINARY_INV,11,2)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    #morph to remove noise
    closing = cv.morphologyEx(threshold, cv.MORPH_CLOSE, kernel)
    dilated = cv.dilate(closing, None, 5)
    return dilated

def resize(img, scale):
    #if image may not change pass
    if scale == 0:
        pass

    #calculate the 50 percent of original dimensions
    width = int(img.shape[1] * scale / 100)
    height = int(img.shape[0] * scale / 100)

    # dsize
    dsize = (width, height)

    # resize image
    output = cv.resize(img, dsize)
    return output

def find_contours(img):
    contours, _ = cv.findContours(img, cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE)
    #print(contours)
    return contours

def draw_lines(img, output):
    lines = cv.HoughLinesP(img,1,np.pi/180,100,minLineLength=100,maxLineGap=12)
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv.line(output,(x1,y1),(x2,y2),(0,255,0),2)
    return output

def biggest_contour(contours):
    cnt = contours[0]
    max_area = cv.contourArea(cnt)
    for cont in contours:
        area = cv.contourArea(cont)
        if area > max_area:
            cnt = cont
            max_area = cv.contourArea(cont)
    return cnt

def approx(img):
    cnt = biggest_contour(img)
    epsilon = 0.01*cv.arcLength(cnt,True)
    approx = cv.approxPolyDP(cnt,epsilon,True)
    return approx

def cut_sudoku(input_img, points):
    try:
        width, height = 1152, 1152
        src = np.float32([points[0],points[1],points[2],points[3]])
        pts2 = np.float32([[height,0],[0,0],[0, width],[height, width]])
        matrix = cv.getPerspectiveTransform(src, pts2)
        result = cv.warpPerspective(input_img, matrix, (width, height))
        return result
    except:
        pass

def preprocess_box(box):
    gray = cv.cvtColor(box, cv.COLOR_RGB2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)
    return thresh







