import os
import sys
import cv2
import csv
import shutil
from tqdm import tqdm
import textwrap
from datetime import datetime

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
import django
django.setup()

from insightface.app import FaceAnalysis
from django.conf import settings

# === Folder Setup ===
BASE = os.path.join(settings.BASE_DIR, "media", "student_faces")
REQUIRED_DIRS = [
    "Original", "640x640", "800x800", "1024x1024", "2048x2048", "bad", "previews", "legend_reference"
]

for subfolder in REQUIRED_DIRS:
    os.makedirs(os.path.join(BASE, subfolder), exist_ok=True)

ORIGINAL_DIR = os.path.join(BASE, "Original")
PREVIEW_DIR = os.path.join(BASE, "previews")
LOG_FILE = os.path.join(BASE, "sorted_face_log.csv")

# Face Analyzer
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))

# Blur detection
def get_sharpness_score(gray_img):
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()

# Hint generator
def generate_hint(w, h, score, sharpness):
    hints = []

    if w < 160 or h < 160:
        hints.append("face too small; retake closer")

    if score < 0.6:
        hints.append("very low detection score; improve lighting")
    elif score < 0.75:
        hints.append("low score; try frontal face & better lighting")
    elif score < 0.85:
        hints.append("acceptable score; frontal & higher res helps")
    else:
        hints.append("good quality")

    aspect_ratio = w / max(h, 1)
    if aspect_ratio < 0.75 or aspect_ratio > 1.5:
        hints.append("non-frontal or tilted face")

    if sharpness < 50:
        hints.append("image blurry; stabilize camera")
    elif sharpness < 100:
        hints.append("slightly soft image; avoid motion blur")
    else:
        hints.append("sharp image")

    return "; ".join(hints)

# === Start Processing ===
with open(LOG_FILE, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])

    # for filename in tqdm(os.listdir(ORIGINAL_DIR)):
    for filename in tqdm(os.listdir(ORIGINAL_DIR), dynamic_ncols=True, leave=False):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        src_path = os.path.join(ORIGINAL_DIR, filename)
        img = cv2.imread(src_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = round(get_sharpness_score(gray), 2)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb)

        if not faces:
            writer.writerow([filename, "-", "-", "-", sharpness, "bad", "no face detected"])
            shutil.copy(src_path, os.path.join(BASE, "bad", filename))
            continue

        face = faces[0]
        bbox = face.bbox.astype(int)
        score = round(face.det_score, 3)

        x1, y1, x2, y2 = bbox
        face_crop = img[y1:y2, x1:x2]
        h, w = y2 - y1, x2 - x1

        if h < 160 or w < 160:
            det = "bad"
        elif max(h, w) <= 320:
            det = "640x640"
        elif max(h, w) <= 480:
            det = "800x800"
        elif max(h, w) <= 700:
            det = "1024x1024"
        else:
            det = "2048x2048"

        hint = generate_hint(w, h, score, sharpness)

        if det == "bad":
            shutil.copy(src_path, os.path.join(BASE, "bad", filename))
        else:
            dest_path = os.path.join(BASE, det, filename)
            cv2.imwrite(dest_path, face_crop)

        writer.writerow([filename, w, h, score, sharpness, det, hint])

        # === Save preview image with overlays ===
        preview_img = img.copy()
        color = (0, 0, 255) if det == "bad" else \
                (128, 0, 128) if score < 0.60 else \
                (0, 255, 255) if score < 0.70 else \
                (0, 165, 255) if score < 0.80 else \
                (0, 255, 0) if score < 0.90 else \
                (255, 0, 0)

        label_score = f"{score:.3f}"
        label_detset = f"Set: {det}"
        label_size = f"Size: {w}x{h}"
        label_sharp = f"Sharp: {sharpness}"

        cv2.rectangle(preview_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(preview_img, label_score, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(preview_img, label_detset, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(preview_img, label_size, (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(preview_img, label_sharp, (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Bottom-left hint
        wrapped_hint = textwrap.wrap(hint, width=60)
        for i, line in enumerate(wrapped_hint):
            cv2.putText(preview_img, line, (10, preview_img.shape[0] - 60 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        cv2.imwrite(os.path.join(PREVIEW_DIR, filename), preview_img)

# === Copy legend image ===
legend_src = os.path.join(BASE, "legend_reference", "legend.jpg")
legend_dest = os.path.join(PREVIEW_DIR, "legend.jpg")

if os.path.exists(legend_src):
    shutil.copy(legend_src, legend_dest)
    print(f"[INFO] Legend image copied to: {legend_dest}")
else:
    print(f"[WARNING] Legend image not found at: {legend_src}")





# import os
# import sys
# import cv2
# import csv
# import shutil
# from tqdm import tqdm
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # === Folder Setup ===
# BASE = os.path.join(settings.BASE_DIR, "media", "student_faces")
# REQUIRED_DIRS = [
#     "Original", "640x640", "800x800", "1024x1024", "2048x2048", "bad", "previews"
# ]
#
# for subfolder in REQUIRED_DIRS:
#     path = os.path.join(BASE, subfolder)
#     os.makedirs(path, exist_ok=True)
#
# ORIGINAL_DIR = os.path.join(BASE, "Original")
# PREVIEW_DIR = os.path.join(BASE, "previews")
# LOG_FILE = os.path.join(BASE, "sorted_face_log.csv")
#
# # Face Analyzer
# face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))
#
# # === Start Processing ===
# with open(LOG_FILE, mode='w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["filename", "face_w", "face_h", "score", "recommended_det_set"])
#
#     for filename in tqdm(os.listdir(ORIGINAL_DIR)):
#         if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#
#         src_path = os.path.join(ORIGINAL_DIR, filename)
#         img = cv2.imread(src_path)
#         if img is None:
#             continue
#
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         if not faces:
#             print(f"[SKIP] No face: {filename}")
#             writer.writerow([filename, "-", "-", "-", "bad"])
#             shutil.copy(src_path, os.path.join(BASE, "bad", filename))
#             continue
#
#         face = faces[0]
#         bbox = face.bbox.astype(int)
#         score = round(face.det_score, 3)
#
#         x1, y1, x2, y2 = bbox
#         face_crop = img[y1:y2, x1:x2]
#         h, w = y2 - y1, x2 - x1
#
#         # Decide det_set by face size
#         if h < 160 or w < 160:
#             det = "bad"
#         elif max(h, w) <= 320:
#             det = "640x640"
#         elif max(h, w) <= 480:
#             det = "800x800"
#         elif max(h, w) <= 700:
#             det = "1024x1024"
#         else:
#             det = "2048x2048"
#
#         # Save face crop or fallback
#         if det == "bad":
#             print(f"[WARN] Small face: {filename}")
#             shutil.copy(src_path, os.path.join(BASE, "bad", filename))
#         else:
#             dest_path = os.path.join(BASE, det, filename)
#             cv2.imwrite(dest_path, face_crop)
#
#         writer.writerow([filename, w, h, score, det])
#
#         # === Save preview image with color, score, det_set, and face size ===
#         preview_img = img.copy()
#         label_score = f"{score:.3f}"
#         label_detset = f"Set: {det}" if det != "bad" else "Set: BAD"
#         label_size = f"Size: {w}x{h}"
#
#         # Determine color
#         if det == "bad":
#             color = (0, 0, 255)  # Red
#         elif score < 0.60:
#             color = (128, 0, 128)  # Purple
#         elif score < 0.70:
#             color = (0, 255, 255)  # Yellow
#         elif score < 0.80:
#             color = (0, 165, 255)  # Orange
#         elif score < 0.90:
#             color = (0, 255, 0)  # Green
#         else:
#             color = (255, 0, 0)  # Blue
#
#         # Draw bounding box
#         cv2.rectangle(preview_img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(preview_img, label_score, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
#         cv2.putText(preview_img, label_detset, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#         cv2.putText(preview_img, label_size, (x1, y2 + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#
#         # Save preview
#         cv2.imwrite(os.path.join(PREVIEW_DIR, filename), preview_img)
#
#         legend_src = os.path.join(BASE, "legend_reference", "legend.jpg")
#         legend_dest = os.path.join(PREVIEW_DIR, "legend.jpg")
#         shutil.copy(legend_src, legend_dest)
#
#
#         # # Save preview image with box + score
#         # preview_img = img.copy()
#         # label = f"{score:.3f}"
#         # color = (0, 255, 0) if det != "bad" else (0, 0, 255)
#         # cv2.rectangle(preview_img, (x1, y1), (x2, y2), color, 2)
#         # cv2.putText(preview_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
#         # cv2.imwrite(os.path.join(PREVIEW_DIR, filename), preview_img)





# import os
# import sys
# import cv2
# import csv
# import shutil
# from tqdm import tqdm
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # === Config ===
# ORIGINAL_DIR = os.path.join(settings.BASE_DIR, "media/student_faces/Original")
# DEST_BASE = os.path.join(settings.BASE_DIR, "media/student_faces")
# LOG_FILE = os.path.join(settings.BASE_DIR, "media/student_faces/sorted_face_log.csv")
#
# # Ensure destination dirs exist
# DET_SETS = ["640x640", "800x800", "1024x1024", "2048x2048"]
# for d in DET_SETS + ["bad"]:
#     os.makedirs(os.path.join(DEST_BASE, d), exist_ok=True)
#
# # Face Analyzer
# face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))
#
# # Open CSV log
# with open(LOG_FILE, mode='w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["filename", "face_w", "face_h", "score", "recommended_det_set"])
#
#     for filename in tqdm(os.listdir(ORIGINAL_DIR)):
#         if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#
#         src_path = os.path.join(ORIGINAL_DIR, filename)
#         img = cv2.imread(src_path)
#         if img is None:
#             continue
#
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         if not faces:
#             print(f"[SKIP] No face: {filename}")
#             writer.writerow([filename, "-", "-", "-", "bad"])
#             shutil.copy(src_path, os.path.join(DEST_BASE, "bad", filename))
#             continue
#
#         face = faces[0]
#         bbox = face.bbox.astype(int)
#         score = round(face.det_score, 3)
#
#         x1, y1, x2, y2 = bbox
#         face_crop = img[y1:y2, x1:x2]
#         h, w = y2 - y1, x2 - x1
#
#         # Decide det_set by face size
#         if h < 160 or w < 160:
#             det = "bad"
#         elif max(h, w) <= 320:
#             det = "640x640"
#         elif max(h, w) <= 480:
#             det = "800x800"
#         elif max(h, w) <= 700:
#             det = "1024x1024"
#         else:
#             det = "2048x2048"
#
#         # Save crop
#         if det == "bad":
#             print(f"[WARN] Small face: {filename}")
#             shutil.copy(src_path, os.path.join(DEST_BASE, "bad", filename))
#             writer.writerow([filename, w, h, score, "bad"])
#         else:
#             dest_path = os.path.join(DEST_BASE, det, filename)
#             cv2.imwrite(dest_path, face_crop)
#             writer.writerow([filename, w, h, score, det])
