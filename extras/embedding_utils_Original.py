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









# import os
# import cv2
# import numpy as np
# import pickle
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from attendance.models import Student
#
# def run_embedding_on_paths(paths=None, det_set="auto", force=False):
#     EMBEDDING_DIR = os.path.join(settings.MEDIA_ROOT, "embeddings")
#     PKL_PATH = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#     os.makedirs(EMBEDDING_DIR, exist_ok=True)
#
#     if det_set == "auto":
#         det_size = (1024, 1024)
#     else:
#         parts = det_set.split(",")
#         det_size = (int(parts[0]), int(parts[1]))
#
#     debug = getattr(settings, "DEBUG_LOG_EMBEDDINGS", False)
#
#     if debug:
#         print(f"🚀 Embedding generation started")
#         print(f"📂 Input paths: {paths}")
#         print(f"💾 Output folder: {EMBEDDING_DIR}")
#         print(f"📐 Detection size: {det_size}")
#         print(f"🔁 Force mode: {'ON' if force else 'OFF'}")
#
#     app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#     app.prepare(ctx_id=0, det_size=det_size)
#
#     embeddings_dict = {}
#     total_saved = 0
#     total_skipped = 0
#
#     # Collect all image paths recursively from input
#     valid_images = []
#     if paths:
#         for path in paths:
#             if os.path.isfile(path) and path.lower().endswith((".jpg", ".jpeg", ".png")):
#                 valid_images.append(path)
#             elif os.path.isdir(path):
#                 for root, _, files in os.walk(path):
#                     for file in files:
#                         if file.lower().endswith((".jpg", ".jpeg", ".png")):
#                             valid_images.append(os.path.join(root, file))
#
#     # Convert to lower-case path map for fast matching
#     lower_path_map = {os.path.basename(p).lower(): p for p in valid_images}
#
#     for student in Student.objects.all():
#         npy_path = os.path.join(EMBEDDING_DIR, f"{student.h_code}.npy")
#
#         if os.path.exists(npy_path):
#             if force:
#                 os.remove(npy_path)
#                 if debug:
#                     print(f"🗑️ Deleted existing .npy for {student.h_code}")
#             else:
#                 if debug:
#                     print(f"⏩ Skipping {student.h_code}: .npy already exists.")
#                 continue
#
#         face_vectors = []
#
#         for file_path in valid_images:
#             fname = os.path.basename(file_path)
#             if student.h_code.lower() in fname.lower():
#                 img = cv2.imread(file_path)
#                 if img is None:
#                     continue
#                 faces = app.get(img)
#                 if faces:
#                     face_vectors.append(faces[0].embedding)
#
#         if debug:
#             print(f"{student.h_code}: {len(face_vectors)} image(s) used")
#
#         if face_vectors:
#             avg_embedding = np.mean(face_vectors, axis=0)
#             np.save(npy_path, avg_embedding)
#             embeddings_dict[student.h_code] = {
#                 "embedding": avg_embedding,
#                 "name": student.full_name
#             }
#             total_saved += 1
#             if debug:
#                 print(f"✅ Saved .npy for {student.h_code}")
#         else:
#             total_skipped += 1
#             if debug:
#                 print(f"⚠️ No valid faces found for {student.h_code}. Skipped.")
#
#     with open(PKL_PATH, "wb") as f:
#         pickle.dump(embeddings_dict, f)
#
#     if debug:
#         print(f"✅ Saved fallback embeddings to: {PKL_PATH}")
#         print(f"📊 Summary:")
#         print(f"✅ Total students saved: {total_saved}")
#         print(f"⚠️ Total students skipped (no valid images): {total_skipped}")
#
#     return {
#         "saved": total_saved,
#         "skipped": total_skipped,
#         "pkl": PKL_PATH
#     }
#




# import os
# import cv2
# import numpy as np
# import pickle
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from attendance.models import Student
#
# def run_embedding_on_paths(paths=None, det_set="auto", force=False):
#     IMG_FOLDER = os.path.join(settings.MEDIA_ROOT, "student_faces", "face_images")
#     EMBEDDING_DIR = os.path.join(settings.MEDIA_ROOT, "embeddings")
#     PKL_PATH = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#     os.makedirs(EMBEDDING_DIR, exist_ok=True)
#
#     if det_set == "auto":
#         det_size = (1024, 1024)
#     else:
#         parts = det_set.split(",")
#         det_size = (int(parts[0]), int(parts[1]))
#
#     debug = getattr(settings, "DEBUG_LOG_EMBEDDINGS", False)
#
#     if debug:
#         print(f"🚀 Embedding generation started")
#         print(f"📂 Source folder: {IMG_FOLDER}")
#         print(f"💾 Output folder: {EMBEDDING_DIR}")
#         print(f"📐 Detection size: {det_size}")
#         print(f"🔁 Force mode: {'ON' if force else 'OFF'}")
#
#     app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#     app.prepare(ctx_id=0, det_size=det_size)
#
#     embeddings_dict = {}
#     total_saved = 0
#     total_skipped = 0
#
#     selected_filenames = set()
#     if paths:
#         for p in paths:
#             filename = os.path.basename(p)
#             selected_filenames.add(filename.lower())
#
#     for student in Student.objects.all():
#         npy_path = os.path.join(EMBEDDING_DIR, f"{student.h_code}.npy")
#
#         if os.path.exists(npy_path):
#             if force:
#                 os.remove(npy_path)
#                 if debug:
#                     print(f"🗑️ Deleted existing .npy for {student.h_code}")
#             else:
#                 if debug:
#                     print(f"⏩ Skipping {student.h_code}: .npy already exists.")
#                 continue
#
#         face_vectors = []
#
#         for fname in os.listdir(IMG_FOLDER):
#             if selected_filenames and fname.lower() not in selected_filenames:
#                 continue
#             if student.h_code.lower() in fname.lower():
#                 path = os.path.join(IMG_FOLDER, fname)
#                 img = cv2.imread(path)
#                 if img is None:
#                     continue
#                 faces = app.get(img)
#                 if faces:
#                     face_vectors.append(faces[0].embedding)
#
#         if debug:
#             print(f"{student.h_code}: {len(face_vectors)} images used")
#
#         if face_vectors:
#             avg_embedding = np.mean(face_vectors, axis=0)
#             np.save(npy_path, avg_embedding)
#             embeddings_dict[student.h_code] = {
#                 "embedding": avg_embedding,
#                 "name": student.full_name
#             }
#             total_saved += 1
#             if debug:
#                 print(f"✅ Saved .npy for {student.h_code}")
#         else:
#             total_skipped += 1
#             if debug:
#                 print(f"⚠️ No valid faces found for {student.h_code}. Skipped.")
#
#     with open(PKL_PATH, "wb") as f:
#         pickle.dump(embeddings_dict, f)
#
#     if debug:
#         print(f"✅ Saved fallback embeddings to: {PKL_PATH}")
#         print(f"📊 Summary:")
#         print(f"✅ Total students saved: {total_saved}")
#         print(f"⚠️ Total students skipped (no valid images): {total_skipped}")
#
#     return {
#         "saved": total_saved,
#         "skipped": total_skipped,
#         "pkl": PKL_PATH
#     }








# import os
# import cv2
# import numpy as np
# import pickle
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from attendance.models import Student
#
# def run_embedding_on_paths(paths=None, det_set="auto", force=False):
#     IMG_FOLDER = os.path.join(settings.MEDIA_ROOT, "student_faces", "face_images")
#     EMBEDDING_DIR = os.path.join(settings.MEDIA_ROOT, "embeddings")
#     PKL_PATH = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#     os.makedirs(EMBEDDING_DIR, exist_ok=True)
#
#     if det_set == "auto":
#         det_size = (640, 640)  # fallback
#     else:
#         parts = det_set.split(",")
#         det_size = (int(parts[0]), int(parts[1]))
#
#     app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#     app.prepare(ctx_id=0, det_size=det_size)
#
#     embeddings_dict = {}
#     total_saved = 0
#     total_skipped = 0
#
#     students = Student.objects.all()
#
#     for student in students:
#         npy_path = os.path.join(EMBEDDING_DIR, f"{student.h_code}.npy")
#
#         if os.path.exists(npy_path):
#             if force:
#                 os.remove(npy_path)
#             else:
#                 continue
#
#         face_vectors = []
#
#         for fname in os.listdir(IMG_FOLDER):
#             if student.h_code.lower() in fname.lower():
#                 path = os.path.join(IMG_FOLDER, fname)
#                 img = cv2.imread(path)
#                 if img is None:
#                     continue
#                 faces = app.get(img)
#                 if faces:
#                     face_vectors.append(faces[0].embedding)
#
#         if face_vectors:
#             avg_embedding = np.mean(face_vectors, axis=0)
#             np.save(npy_path, avg_embedding)
#             embeddings_dict[student.h_code] = {
#                 "embedding": avg_embedding,
#                 "name": student.full_name
#             }
#             total_saved += 1
#         else:
#             total_skipped += 1
#
#     with open(PKL_PATH, "wb") as f:
#         pickle.dump(embeddings_dict, f)
#
#     return {
#         "saved": total_saved,
#         "skipped": total_skipped,
#         "pkl": PKL_PATH
#     }












# import os
# import cv2
# from insightface.app import FaceAnalysis
# import numpy as np
# import pickle
# from django.conf import settings
# from attendance.models import Student
#
# def run_embedding_on_paths(paths, det_set="auto"):
#     all_images = []
#
#     # Collect all images recursively
#     for path in paths:
#         if os.path.isfile(path) and path.lower().endswith((".jpg", ".jpeg", ".png")):
#             all_images.append(path)
#         elif os.path.isdir(path):
#             for root, _, files in os.walk(path):
#                 for f in files:
#                     if f.lower().endswith((".jpg", ".jpeg", ".png")):
#                         all_images.append(os.path.join(root, f))
#
#     if not all_images:
#         raise Exception("No valid image files found.")
#
#     if det_set == "auto":
#         det_size = (800, 800)  # Default fallback
#     else:
#         parts = det_set.split(",")
#         det_size = (int(parts[0]), int(parts[1]))
#
#     # Load model
#     face_analyzer = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
#     face_analyzer.prepare(ctx_id=0, det_size=det_size)
#
#     embedding_dir = os.path.join(settings.MEDIA_ROOT, "embeddings")
#     os.makedirs(embedding_dir, exist_ok=True)
#     embeddings_map = {}
#
#     for image_path in all_images:
#         img = cv2.imread(image_path)
#         if img is None:
#             continue
#
#         faces = face_analyzer.get(img)
#         if not faces:
#             continue
#
#         face = faces[0]
#         if face.embedding is None:
#             continue
#
#         # Infer student from filename (H-CODE-XX.jpg)
#         # filename = os.path.basename(image_path)
#         # h_code = filename.split("-")[0]
#         filename = os.path.basename(image_path)
#         name, _ = os.path.splitext(filename)
#         h_code = name.split("-")[0].split("_")[0]  # handles both - and _ separators
#
#         embeddings_map[h_code] = face.embedding
#
#         # Save individual .npy
#         np.save(os.path.join(embedding_dir, f"{h_code}.npy"), face.embedding)
#
#     # Save combined pkl
#     pkl_path = os.path.join(embedding_dir, "face_embeddings.pkl")
#     with open(pkl_path, "wb") as f:
#         pickle.dump(embeddings_map, f)
#
#     return len(embeddings_map)
