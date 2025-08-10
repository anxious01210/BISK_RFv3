# extras/embedding_utils.py:

import os
import cv2
import numpy as np
import pickle
from django.conf import settings
from insightface.app import FaceAnalysis
from attendance.models import Student

def run_embedding_on_paths(paths=None, det_set="auto", force=False):
    EMBEDDING_DIR = os.path.join(settings.MEDIA_ROOT, "embeddings")
    PKL_PATH = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    if det_set == "auto":
        det_size = (1024, 1024)
    else:
        parts = det_set.split(",")
        det_size = (int(parts[0]), int(parts[1]))

    debug = getattr(settings, "DEBUG_LOG_EMBEDDINGS", False)

    if debug:
        print(f"🚀 Embedding generation started")
        print(f"📂 Input paths: {paths}")
        print(f"💾 Output folder: {EMBEDDING_DIR}")
        print(f"📐 Detection size: {det_size}")
        print(f"🔁 Force mode: {'ON' if force else 'OFF'}")

    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=det_size)

    embeddings_dict = {}
    total_saved = 0
    total_skipped = 0

    # Collect all image paths recursively from input
    def collect_images(paths):
        valid_images = []
        for path in paths:
            if os.path.isfile(path) and path.lower().endswith((".jpg", ".jpeg", ".png")):
                valid_images.append(path)
            elif os.path.isdir(path):
                for root, _, files in os.walk(path):
                    for file in files:
                        if file.lower().endswith((".jpg", ".jpeg", ".png")):
                            valid_images.append(os.path.join(root, file))
        return valid_images

    valid_images = collect_images(paths or [])

    # Convert to lower-case path map for fast matching
    lower_path_map = {os.path.basename(p).lower(): p for p in valid_images}

    for student in Student.objects.all():
        npy_path = os.path.join(EMBEDDING_DIR, f"{student.h_code}.npy")

        if os.path.exists(npy_path):
            if force:
                os.remove(npy_path)
                if debug:
                    print(f"🗑️ Deleted existing .npy for {student.h_code}")
            else:
                if debug:
                    print(f"⏩ Skipping {student.h_code}: .npy already exists.")
                continue

        face_vectors = []

        for file_path in valid_images:
            fname = os.path.basename(file_path)
            if student.h_code.lower() in fname.lower():
                img = cv2.imread(file_path)
                if img is None:
                    continue
                faces = app.get(img)
                if faces:
                    face_vectors.append(faces[0].embedding)

        if debug:
            print(f"{student.h_code}: {len(face_vectors)} image(s) used")

        if face_vectors:
            avg_embedding = np.mean(face_vectors, axis=0)
            np.save(npy_path, avg_embedding)
            embeddings_dict[student.h_code] = {
                "embedding": avg_embedding,
                "name": student.full_name
            }
            total_saved += 1
            if debug:
                print(f"✅ Saved .npy for {student.h_code}")
        else:
            total_skipped += 1
            if debug:
                print(f"⚠️ No valid faces found for {student.h_code}. Skipped.")

    with open(PKL_PATH, "wb") as f:
        pickle.dump(embeddings_dict, f)

    if debug:
        print(f"✅ Saved fallback embeddings to: {PKL_PATH}")
        print(f"📊 Summary:")
        print(f"✅ Total students saved: {total_saved}")
        print(f"⚠️ Total students skipped (no valid images): {total_skipped}")

    return {
        "saved": total_saved,
        "skipped": total_skipped,
        "pkl": PKL_PATH
    }


