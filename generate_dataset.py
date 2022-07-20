from operator import ge
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageChops
import numpy as np
import pathlib
import os
from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
import cv2 as cv

characters = ['1','2','3','4','5','6','7','8','9']
IMG_WIDTH, IMG_HEIGHT = 100, 100
count = 0

def cb(img):
    img = np.array(img)
    mean = img.mean()
    non_empty_columns = np.where(img.min(axis=0)<mean)[0]
    non_empty_rows = np.where(img.min(axis=1)<mean)[0]
    boundingBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
    bb = boundingBox
    return bb


def draw_char(font_type, number, path):
    background_color = (255)
    foreground_color = (0)
    image = Image.new("L", (IMG_WIDTH, IMG_HEIGHT), background_color) 
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(font_type, 50)
    w, h = draw.textsize(number, font=font)
    draw.text(((IMG_WIDTH - w) / 2, (IMG_HEIGHT - h) / 2), number, (foreground_color), font=font)
    blurred = image.filter(filter=ImageFilter.GaussianBlur(1))
    bb = cb(blurred)
    blurred = np.array(blurred)[bb[0]:bb[1], bb[2]:bb[3]]
    blurred = cv.resize(blurred,(28,28))
    cv.imwrite(f"{path}/{count}.jpg", blurred)


def supportsDigits(fontPath):
    font = TTFont(fontPath)
    required_ordinals = [ord(glyph) for glyph in ['1','2','3','4','5','6','7','8','9']]
    for table in font['cmap'].tables:
        for o in required_ordinals:
            if not o in table.cmap.keys():
                return False
    return True

font_blacklist = ["/usr/share/fonts/truetype/lyx/esint10.ttf"]

def check_blacklist(font):
    for bl in font_blacklist:
        if bl in font:
            return False
    return True

with open("fonts.list", "r") as fonts:
    for font in fonts:
        font = font.strip()
        can_use = supportsDigits(font)
        if can_use and check_blacklist(font):
            for character in characters:
                path = f'testing_data/{character}' if (count % 10) == 0 else f'training_data/{character}'
                pathlib.Path(path).mkdir(parents=True, exist_ok=True)
                print(font)
                draw_char(font, character, path)
                count = count+1





