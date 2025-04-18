# face_recognition/detector.py
from retinaface import RetinaFace
import cv2
import numpy as np

# Initialize the face detector
detector = RetinaFace(quality='normal')  # or quality='high' for best accuracy

def detect_faces(image_path_or_array):
    if isinstance(image_path_or_array, str):
        img = cv2.imread(image_path_or_array)
    else:
        img = image_path_or_array

    faces = RetinaFace.detect_faces(img)
    return faces
