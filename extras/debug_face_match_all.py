import os
import sys
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
import django
django.setup()

from insightface.app import FaceAnalysis
from django.conf import settings

# === CONFIG ===
TEST_IMAGE = os.path.join(settings.BASE_DIR, 'media/student_faces/H123456_9.jpg')  # simulate live frame
EMBEDDING_DIR = os.path.join(settings.BASE_DIR, 'media/embeddings')  # path to *.npy files
TOP_N = 5  # top matches to display

# === Initialize Face Analyzer ===
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))  # match recognition setting
face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))  # match recognition setting
# face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))  # match recognition setting

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

# Use first detected face
face = faces[0]
bbox = face.bbox.astype(int)
embedding = face.embedding
det_score = face.det_score

if embedding is None:
    print("[ERROR] No embedding generated.")
    sys.exit(1)

# Crop and save face for inspection
face_crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
crop_path = os.path.join(settings.BASE_DIR, 'media/debug_crop.jpg')
cv2.imwrite(crop_path, face_crop)

# === Print detection info ===
print("\n=== FACE INFO ===")
print(f"Detection Score: {det_score:.3f}")
print(f"Face Size: {(bbox[3]-bbox[1])}x{(bbox[2]-bbox[0])} pixels")
print(f"Embedding Norm: {np.linalg.norm(embedding):.4f}")
print(f"Cropped face saved to: {crop_path}")

# === Load and compare all embeddings ===
print("\n=== TOP MATCHES ===")
results = []
for npy_file in glob(os.path.join(EMBEDDING_DIR, "*.npy")):
    h_code = os.path.splitext(os.path.basename(npy_file))[0]
    known_embedding = np.load(npy_file)

    try:
        score = cosine_similarity([embedding], [known_embedding])[0][0]
        results.append((h_code, score))
    except Exception as e:
        print(f"[ERROR] Failed comparing with {h_code}: {e}")

# Sort and show top N
results.sort(key=lambda x: x[1], reverse=True)
for rank, (h_code, score) in enumerate(results[:TOP_N], start=1):
    color = (
        "🟥" if score < 0.50 else
        "🟨" if score < 0.80 else
        "🟩"
    )
    print(f"{color} [{rank}] {h_code}: Score = {score:.4f}")

if not results:
    print("[WARNING] No embeddings matched or loaded.")
