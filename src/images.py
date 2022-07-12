import cv2 as cv
import numpy as np
import utlis
import math
from utlis import *

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

def angle_between(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector2 = vector_2 / np.linalg.norm(vector_2)
    dot_droduct = np.dot(unit_vector_1, unit_vector2)
    angle = np.arccos(dot_droduct)
    return angle * 57.2958 

def side_lengths_are_too_different(A, B, C, D, eps_scale):
    AB = math.sqrt((A[0]-B[0])**2 + (A[1]-B[1])**2)
    AD = math.sqrt((A[0]-D[0])**2 + (A[1]-D[1])**2)
    BC = math.sqrt((B[0]-C[0])**2 + (B[1]-C[1])**2)
    CD = math.sqrt((C[0]-D[0])**2 + (C[1]-D[1])**2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > eps_scale * shortest

def find_contours(img):
    contours, _ = cv.findContours(img, cv.RETR_TREE,
            cv.CHAIN_APPROX_SIMPLE)
    #print(contours)
    return contours

def approx_90_degrees(angle, epsilon):
    return abs(angle - 90) < epsilon


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
    if len(approx)  == 4:
        return approx
    return None

def cut_sudoku(input_img, points):
    width, height = 1152, 1152
    src = np.float32([*points])
    pts2 = np.float32([[height,0],[0,0],[0, width],[height, width]])
    matrix = cv.getPerspectiveTransform(src, pts2)
    result = cv.warpPerspective(input_img, matrix, (width, height))
    return result
    
def preprocess_box(box):
    gray = cv.cvtColor(box, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray,(9,9),0)
    th3 = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,7,2)
    return th3


def prepare_box(img):
    img = np.array(img)
    img = img[5:]
    img = img[:-5]
    img = np.delete(img, range(0,5), 1)
    img = np.delete(img, range(-5, 0), 1)

    # Top
    while img[0].mean() <= 230:
        img = img[1:]

    # Down
    while img[-1].mean() <= 230:
        img = img[:-1]

    #Left
    while img[:,0].mean() <= 230:
        img = np.delete(img, 0, 1)

    #Right
    while img[:, -1].mean() <= 230:
        img = np.delete(img, -1, 1)

    mean = img.mean()
    img = img[5:]
    img = img[:-5]
    img = np.delete(img, range(0,5), 1)
    img = np.delete(img, range(-5, 0), 1)

    if mean > 250:
        return cv.resize(img, (28, 28))

    non_empty_columns = np.where(img.min(axis=1)<mean)[0]
    non_empty_rows = np.where(img.min(axis=1)<mean)[0]
    bb = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    img = cv.bitwise_not(img)
    img = img[bb[0]:bb[1], bb[2]:bb[3]] 
    img = cv.resize(img, (28, 28))
    
    return img

def overlay(img_out, img_solved, biggest, w, h):
    print(img_out.dtype)
    print(img_solved.dtype)
    img_solved = img_solved.astype('uint8')
    pts2 = np.float32(biggest) 
    pts1 =  np.float32([[1152, 0],[0, 0], [0, 1152],[1152, 1152]]) 
    matrix = cv.getPerspectiveTransform(pts1, pts2)  
    imgInvimgColored = img_solved.copy()
    imgInvimgColored = cv.imgPerspective(img_solved, matrix, (w, h))
    print(imgInvimgColored.shape)
    print(img_out.shape)    
    inv_perspective = cv.addWeighted(imgInvimgColored, 1, img_out, 0.5, 1)
    return inv_perspective
    
def largest_connected_component(image):

    image = image.astype('uint8')
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[:, -1]

    if(len(sizes) <= 1):
        blank_image = np.zeros(image.shape)
        blank_image.fill(255)
        return blank_image

    max_label = 1
    # Start from component 1 (not 0) because we want to leave out the background
    max_size = sizes[1]     

    for i in range(1, nb_components):
        if sizes[i] > max_size:
            max_label = i
            max_size = sizes[i]

    img2 = np.zeros(output.shape)
    img2.fill(0)
    img2[output == max_label] = 255
    return img2

def recognize_and_solve_sudoku(input_sudoku):
    eps_angle = 20
    out_sudoku = np.zeros((900,900))
    #preprocess current camera frame
    img_preprocessed = preprocess(input_sudoku)
    #find all contours form image
    finded = find_contours(img_preprocessed)
    #find points of corners of sudoku 
    corners = approx(finded)

    if corners is None:
        return input_sudoku
    rect = np.zeros((4, 2), dtype = "float32")
    corners = corners.reshape(4,2)
     # Find top left (sum of coordinates is the smallest)
    sum = 10000
    index = 0
    for i in range(4):
        if(corners[i][0]+corners[i][1] < sum):
            sum = corners[i][0]+corners[i][1]
            index = i
    rect[0] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find bottom right (sum of coordinates is the biggest)
    sum = 0
    for i in range(3):
        if(corners[i][0]+corners[i][1] > sum):
            sum = corners[i][0]+corners[i][1]
            index = i
    rect[2] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find top right (Only 2 points left, should be easy
    if(corners[0][0] > corners[1][0]):
        rect[1] = corners[0]
        rect[3] = corners[1]
        
    else:
        rect[1] = corners[1]
        rect[3] = corners[0]

    A = rect[3]
    B = rect[2]
    C = rect[1]
    D = rect[0]

    AB = B - A      
    AD = D - A
    BC = C - B
    DC = C - D

    rect = rect.reshape(4,2)

    if not (approx_90_degrees(angle_between(AB,AD), eps_angle) and approx_90_degrees(angle_between(AB,BC), eps_angle)
    and approx_90_degrees(angle_between(BC,DC), eps_angle) and approx_90_degrees(angle_between(AD,DC), eps_angle)):
        return input_sudoku

    eps_scale = 1.2     # Longest cannot be longer than epsScale * shortest
    if(side_lengths_are_too_different(A, B, C, D, eps_scale)):
        return input_sudoku

    shrinked_board = cut_sudoku(input_sudoku, rect)

    shrinked_board = cv.flip(shrinked_board, 1)

    boxes = split_photo(shrinked_board)


    prediction_img, predictions, posarr = display_predictions(boxes)

    cv.imwrite("Preds.jpg", prediction_img)

    return prediction_img
