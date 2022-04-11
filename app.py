from flask import Flask,render_template,Response
import cv2 as cv
import numpy as np
import images as im
from utlis import *
import tensorflow as tf
import os
import sudoku

app=Flask(__name__)

def generate_frames():
    camera=cv.VideoCapture(0)
    while True:   
        # read the camera frame
        success,cap=camera.read()
        #preprocess current camera frame
        img_preprocessed = im.preprocess(cap)
        #find all contours form image
        finded = im.find_contours(img_preprocessed)
        #find points of corners of sudoku 
        approx = im.approx(finded)
        #cut sudoku board from photo
        if len(approx) == 4:
            shrinked_board = im.cut_sudoku(cap, approx)
        if not success:
            break   
        else:
            if len(approx) == 4:
                ret,buffer=cv.imencode('.jpg', shrinked_board)
                frame=buffer.tobytes()
            else: 
                ret,buffer=cv.imencode('.jpg',cv.drawContours(cap, approx,-1, (0,255,0), 20))
                frame=buffer.tobytes()


        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)