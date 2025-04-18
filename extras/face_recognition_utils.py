from retinaface import RetinaFace
import cv2

img_path = "test.png"  # Use any clear image of a face
faces = RetinaFace.detect_faces(img_path)
print(faces)
