import cv2 as cv
import numpy as np
import math
from utlis import *
import sudoku


def split_photo(img):
    """
    split sudoku board image into 81
    squres
    """
    vsplit = np.vsplit(img, 9)
    boxes = []
    for vs in vsplit:
        hsplit = np.hsplit(vs, 9)
        for hs in hsplit:
            boxes.append(hs)
    return boxes


def preprocess(img):
    """
    This function preprocess the image.
    It performs basic operations like
    applaying a blur and treshold etc.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # add blur
    gray = cv.GaussianBlur(gray, (9, 9), 0)
    # add bibteralFilter to reduce noise
    frame = cv.bilateralFilter(gray, 9, 75, 75)
    # applay inverted treshold
    threshold = cv.adaptiveThreshold(
        frame, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2
    )
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    # morph to remove noise
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
    AB = math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
    AD = math.sqrt((A[0] - D[0]) ** 2 + (A[1] - D[1]) ** 2)
    BC = math.sqrt((B[0] - C[0]) ** 2 + (B[1] - C[1]) ** 2)
    CD = math.sqrt((C[0] - D[0]) ** 2 + (C[1] - D[1]) ** 2)
    shortest = min(AB, AD, BC, CD)
    longest = max(AB, AD, BC, CD)
    return longest > eps_scale * shortest


def find_contours(img):
    """
    find contours in image
    """
    contours, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return contours


def approx_90_degrees(angle, epsilon):
    return abs(angle - 90) < epsilon


def biggest_contour(contours):
    """
    Find biggest contour on image
    """
    cnt = contours[0]
    max_area = cv.contourArea(cnt)
    for cont in contours:
        area = cv.contourArea(cont)
        if area > max_area:
            cnt = cont
            max_area = cv.contourArea(cont)
    return cnt


def approx(img):
    """
    Find 4 corners of sudoku board
    from image
    """
    cnt = biggest_contour(img)
    epsilon = 0.01 * cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, epsilon, True)
    if len(approx) == 4:
        return approx
    return None


def cut_sudoku(input_img, points):
    """
    Get sudoku board from
    original image
    """
    width, height = 1152, 1152
    src = np.float32([*points])
    pts2 = np.float32([[height, 0], [0, 0], [0, width], [height, width]])
    matrix = cv.getPerspectiveTransform(src, pts2)
    result = cv.warpPerspective(input_img, matrix, (width, height))
    return result


def preprocess_box(box):
    """
    perform some required operation
    on sudoku box before OCR
    """
    gray = cv.cvtColor(box, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    th3 = cv.adaptiveThreshold(
        blur, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 7, 2
    )
    return th3


def boundings(img):
    img = np.array(img)
    mean = img.mean()
    non_empty_columns = np.where(img.min(axis=0) < mean)[0]
    non_empty_rows = np.where(img.min(axis=1) < mean)[0]
    boundingBox = (
        min(non_empty_rows),
        max(non_empty_rows),
        min(non_empty_columns),
        max(non_empty_columns),
    )
    bb = boundingBox
    return bb


def prepare_box(img):
    """
    Cleans sudoku boxes from noise
    before recognition
    """
    img = np.array(img)
    # erode image to connect charagters
    kernel = np.ones((2, 2), np.uint8)
    img = cv.erode(img, kernel, iterations=1)
    # remove a few pixels from evry side of image
    img = img[5:]
    img = img[:-5]
    img = np.delete(img, range(0, 5), 1)
    img = np.delete(img, range(-5, 0), 1)

    # Top
    while img[0].mean() <= 230:
        img = img[1:]

    # Down
    while img[-1].mean() <= 230:
        img = img[:-1]

    # Left
    while img[:, 0].mean() <= 230:
        img = np.delete(img, 0, 1)

    # Right
    while img[:, -1].mean() <= 230:
        img = np.delete(img, -1, 1)

    # Top
    while img[0].mean() <= 230:
        img = img[1:]

    # Down
    while img[-1].mean() <= 230:
        img = img[:-1]

    # Left
    while img[:, 0].mean() <= 230:
        img = np.delete(img, 0, 1)

    # Right
    while img[:, -1].mean() <= 230:
        img = np.delete(img, -1, 1)

    img = img[5:]
    img = img[:-5]
    img = np.delete(img, range(0, 5), 1)
    img = np.delete(img, range(-5, 0), 1)

    # Top
    while img[0].mean() <= 230:
        img = img[1:]

    # Down
    while img[-1].mean() <= 230:
        img = img[:-1]

    # Left
    while img[:, 0].mean() <= 230:
        img = np.delete(img, 0, 1)

    # Right
    while img[:, -1].mean() <= 230:
        img = np.delete(img, -1, 1)

    mean = img.mean()

    if mean > 250:
        return cv.resize(img, (28, 28))

    # find contours in image
    cnts3 = img.copy()
    contours2, hierarchy2 = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    all_areas = []

    contours2 = sorted(contours2, key=cv.contourArea, reverse=True)
    # skip first contour - it is image boundings
    contours2 = contours2[1:]

    for cnt in contours2:
        area = cv.contourArea(cnt)
        all_areas.append(area)

    if len(all_areas) == 0:
        bb = boundings(img)
        img = img[bb[0] : bb[1], bb[2] : bb[3]]
        img = cv.bitwise_not(img)
        img = cv.resize(img, (28, 28))
        return img

    # calculate average contour
    avg_cnt = sum(all_areas) / len(all_areas)

    for cnt in contours2:
        area = cv.contourArea(cnt)
        # if area of contour is smaller remove it
        # becouse it is probably noise
        if area < avg_cnt:
            cv.drawContours(cnts3, [cnt], -1, (255, 255, 255), -1)

    cnts3 = cnts3[5:]
    cnts3 = cnts3[:-5]
    cnts3 = np.delete(cnts3, range(0, 5), 1)
    cnts3 = np.delete(cnts3, range(-5, 0), 1)
    img = cnts3

    bb = boundings(img)
    img = img[bb[0] : bb[1], bb[2] : bb[3]]
    img = cv.bitwise_not(img)
    img = cv.GaussianBlur(img, (3, 3), 0)
    img = cv.resize(img, (28, 28))

    return img


def overlay(img_out, img_solved, biggest, w, h):
    """
    Overlays solved sudoku to original image
    """
    img_solved = img_solved.astype("uint8")
    pts2 = np.float32(biggest)
    pts1 = np.float32([[1152, 0], [0, 0], [0, 1152], [1152, 1152]])
    matrix = cv.getPerspectiveTransform(pts1, pts2)
    imgInvWarpColored = img_solved.copy()
    imgInvWarpColored = cv.warpPerspective(img_solved, matrix, (w, h))
    inv_perspective = cv.addWeighted(imgInvWarpColored, 1, img_out, 0.5, 1)
    return inv_perspective


def recognize_and_solve_sudoku(input_sudoku):
    """
    Main function
    """
    eps_angle = 20
    # preprocess current camera frame
    img_preprocessed = preprocess(input_sudoku)
    # find all contours form image
    finded = find_contours(img_preprocessed)
    # find points of corners of sudoku
    corners = approx(finded)
    aprx = approx(finded)

    if corners is None:
        return input_sudoku
    rect = np.zeros((4, 2), dtype="float32")
    corners = corners.reshape(4, 2)
    # Find top left (sum of coordinates is the smallest)
    sum = 10000
    index = 0

    for i in range(4):
        if corners[i][0] + corners[i][1] < sum:
            sum = corners[i][0] + corners[i][1]
            index = i

    rect[0] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find bottom right (sum of coordinates is the biggest)
    sum = 0
    for i in range(3):
        if corners[i][0] + corners[i][1] > sum:
            sum = corners[i][0] + corners[i][1]
            index = i
    rect[2] = corners[index]
    corners = np.delete(corners, index, 0)

    # Find top right (Only 2 points left, should be easy
    if corners[0][0] > corners[1][0]:
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

    rect = rect.reshape(4, 2)

    if not (
        approx_90_degrees(angle_between(AB, AD), eps_angle)
        and approx_90_degrees(angle_between(AB, BC), eps_angle)
        and approx_90_degrees(angle_between(BC, DC), eps_angle)
        and approx_90_degrees(angle_between(AD, DC), eps_angle)
    ):
        return input_sudoku

    eps_scale = 1.2  # Longest cannot be longer than epsScale * shortest
    if side_lengths_are_too_different(A, B, C, D, eps_scale):
        return input_sudoku

    shrinked_board = cut_sudoku(input_sudoku, rect)

    shrinked_board = cv.flip(shrinked_board, 1)

    boxes = split_photo(shrinked_board)
    prediction_img, predictions, posarr = display_predictions(boxes)

    sudoku.solve(predictions.reshape(9, 9))

    solved_board = np.reshape(predictions, (81)) * posarr

    if solved_board.sum() == 0:
        return input_sudoku

    solved_img, predictions, _ = display_predictions(solved_board, solved=True)

    (h, w) = input_sudoku.shape[:2]

    simg = im.overlay(input_sudoku, solved_img, aprx, w, h)

    return simg
