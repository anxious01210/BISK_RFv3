import os
import cv2
from insightface.app import FaceAnalysis

# img = cv2.imread("media/student_faces/H123456.png")  # Replace with one of your known images
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, "media", "student_faces", "H123456.png")
img = cv2.imread(img_path)

if img is None:
    print("❌ Failed to load image. Check path or file integrity.")
    exit()

recognizer = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
recognizer.prepare(ctx_id=0)

faces = recognizer.get(img)
print(f"Detected {len(faces)} faces.")
if faces:
    for face in faces:
        print("Embedding shape:", face.embedding.shape)
