import cv2 as cv
import numpy as np
import os
import shutil

PROJECT_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(PROJECT_DIR, 'images')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
OUT_DIR = os.path.join(PROJECT_DIR, 'out')

PEOPLE_DIR = os.path.join(OUT_DIR, 'people')
NO_PEOPLE_DIR = os.path.join(OUT_DIR, 'no_people')

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PEOPLE_DIR, exist_ok=True)
os.makedirs(NO_PEOPLE_DIR, exist_ok=True)

cascade_path = os.path.join(MODELS_DIR, 'haarcascade_frontalface_default.xml')

face_cascade =cv.CascadeClassifier(cascade_path)

if face_cascade.empty():
    print('No face cascade detected')
    exit()

def face_detect(img_bgr):
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
    return faces