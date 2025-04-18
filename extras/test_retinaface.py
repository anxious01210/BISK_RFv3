import cv2
from retinaface import RetinaFace
import torch

print("CUDA available:", torch.cuda.is_available())

img_path = "test.png"  # Replace with any image path
faces = RetinaFace.detect_faces(img_path)

if isinstance(faces, dict):
    print("Faces detected:", len(faces))
    for key in faces:
        identity = faces[key]
        facial_area = identity["facial_area"]
        print(f"Face {key}: {facial_area}")
else:
    print("No faces detected or error in detection.")
