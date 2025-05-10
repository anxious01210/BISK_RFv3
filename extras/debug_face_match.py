import os
import sys
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Setup Django and project settings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
import django
django.setup()

from insightface.app import FaceAnalysis
from django.conf import settings

# Load face analyzer
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))  # match recognition det_set
# face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))  # match recognition det_set

# === CONFIG ===
EMBEDDING_NPY = os.path.join(settings.BASE_DIR, 'media/embeddings/H123456.npy')  # path to known embedding
TEST_IMAGE = os.path.join(settings.BASE_DIR, 'media/student_faces/H123456_1.jpg')  # test image to simulate camera

# === Load and analyze test image ===
img = cv2.imread(TEST_IMAGE)
if img is None:
    print(f"[ERROR] Cannot load image: {TEST_IMAGE}")
    sys.exit(1)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
faces = face_analyzer.get(img_rgb)

if not faces:
    print("[WARNING] No face detected.")
    sys.exit(1)

# Assume first face
face = faces[0]
bbox = face.bbox.astype(int)
embedding = face.embedding
det_score = face.det_score

if embedding is None:
    print("[ERROR] No embedding found.")
    sys.exit(1)

# Crop face for inspection
face_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
crop_path = os.path.join(settings.BASE_DIR, 'media/debug_crop.jpg')
cv2.imwrite(crop_path, face_crop)

# Log face info
print("=== FACE INFO ===")
print(f"Detection Score: {det_score:.3f}")
print(f"Face Size: {(bbox[3]-bbox[1])}x{(bbox[2]-bbox[0])} pixels")
print(f"Embedding Norm: {np.linalg.norm(embedding):.4f}")

# Load known embedding
known_embedding = np.load(EMBEDDING_NPY)

# Compute cosine similarity
score = cosine_similarity([embedding], [known_embedding])[0][0]
print(f"Cosine Similarity Score: {score:.4f}")
