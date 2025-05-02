# extras/initialize_embeddings.py
# This script will process each student's images, generate their embeddings, and save them individually as .npy files.
# It also maintains a combined face_embeddings.pkl file for fallback purposes. The --force flag allows you to regenerate
# embeddings for all students, overwriting existing files.
# 1) Generate individual .npy files for each student by averaging the embeddings from all their images.
# 2) Create or update a combined face_embeddings.pkl file that includes embeddings for all students.
# 3) Skip existing .npy files unless the --force flag is used, in which case it regenerates all embeddings.
# Normal mode (only generates missing .npy files):      python3 extras/initialize_embeddings.py
# Force mode (regenerates all):                         python3 extras/initialize_embeddings.py --force

# Recommended det_size Settings:
#     For High-Resolution Inputs (e.g., 4K cameras): Consider increasing det_size to 800x800 or 1080x1080 to capture finer facial details.
#     For Multiple Faces in a Frame: A larger det_size can help in accurately detecting and recognizing multiple faces simultaneously.
#     For Real-Time Processing Needs: Balance between accuracy and speed by selecting a moderate det_size like 800x800.


# Implementation Steps:
#         Update det_size in Your Scripts:
#             In initialize_embeddings.py:
#                 app.prepare(ctx_id=0, det_size=(1080, 1080))
#             In recognize_and_log_attendance_parallel.py:
#                 face_analyzer.prepare(ctx_id=0, det_size=(1080, 1080))

        # Regenerate Embeddings:
        #     After changing det_size, regenerate the embeddings to ensure consistency. Use the --force flag to overwrite existing embeddings:
        #         python3 extras/initialize_embeddings.py --force

        # Test and Monitor Performance:
        #   Evaluate the system's performance with the new det_size. Monitor processing speed and recognition accuracy
        #   to ensure the new settings meet your requirements.



# In InsightFace, the det_size parameter in the prepare() method specifies the dimensions to which the entire input
# image (such as a full video frame) is resized before the face detection process begins. This resizing is crucial
# because it affects the accuracy and speed of face detection:
#     Larger det_size values (e.g., 1080x1080) can improve the detection of small or distant faces by providing more
#     detail, but they require more computational resources and may slow down processing.
#     Smaller det_size values (e.g., 640x640) offer faster processing with less computational demand but may compromise
#     accuracy for small faces.
# After detection, InsightFace crops and aligns the detected face regions to a standard size (typically 112x112 pixels)
# for embedding generation. This standardization ensures consistency in the face recognition process.

# InsightFace doesn’t support any arbitrary size. Valid values depend on the RetinaFace model configuration and must be
# divisible by 32 (since the backbone downsamples the image multiple times). Common and tested safe values include:
#     640 × 640 ← Default and fast, good for near/mid-range faces
#     720 × 720
#     768 × 768
#     800 × 800 ← Good balance of accuracy and performance
#     960 × 960
#     1024 × 1024 ← High accuracy, slower
#     ❌ 1080 × 1080 → This often fails due to shape mismatch (as you just saw)
#     1280 × 1280 ← Max before VRAM becomes a problem (4K+ GPUs only)

# Since you're planning to handle 6 × 4K streams and upgrade your GPU:
#     Use 800×800 for now (you already tested it and got better results)
#     Optionally test 1024×1024 if your GPU (like an RTX 4080+) can handle it



import os
import sys
import cv2
import numpy as np
import pickle
import argparse

# Setup Django
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
import django
django.setup()

from insightface.app import FaceAnalysis
from django.conf import settings
from attendance.models import Student

def main(force=False):
    # Initialize Face Analyzer
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(1024, 1024))

    # Paths
    IMG_FOLDER = os.path.join(settings.BASE_DIR, "media/student_faces")
    EMBEDDING_DIR = os.path.join(settings.BASE_DIR, "media/embeddings")
    PKL_PATH = os.path.join(settings.BASE_DIR, "media/face_embeddings.pkl")
    os.makedirs(EMBEDDING_DIR, exist_ok=True)

    # Counters
    total_saved = 0
    total_skipped = 0

    # Result dictionary
    embeddings_dict = {}

    # Process each student
    for student in Student.objects.all():
        npy_path = os.path.join(EMBEDDING_DIR, f"{student.h_code}.npy")

        if os.path.exists(npy_path):
            if force:
                os.remove(npy_path)
                print(f"🗑️ Deleted existing .npy for {student.h_code}")
            else:
                print(f"⏩ Skipping {student.h_code}: .npy already exists.")
                continue

        face_vectors = []
        for file in os.listdir(IMG_FOLDER):
            if student.h_code.lower() in file.lower():
                path = os.path.join(IMG_FOLDER, file)
                img = cv2.imread(path)
                if img is None:
                    continue
                faces = app.get(img)
                if faces:
                    face_vectors.append(faces[0].embedding)

        print(f"{student.h_code}: {len(face_vectors)} images used")

        if face_vectors:
            avg_embedding = np.mean(face_vectors, axis=0)
            embeddings_dict[student.h_code] = avg_embedding
            np.save(npy_path, avg_embedding)
            total_saved += 1
            print(f"✅ Saved .npy for {student.h_code}")
        else:
            total_skipped += 1
            print(f"⚠️ No valid faces found for {student.h_code}. Skipped.")

    # Save fallback .pkl
    with open(PKL_PATH, 'wb') as f:
        pickle.dump(embeddings_dict, f)
    print(f"✅ Saved fallback embeddings to: {PKL_PATH}")

    # Summary
    print(f"\n📊 Summary:")
    print(f"✅ Total students saved: {total_saved}")
    print(f"⚠️ Total students skipped (no valid images): {total_skipped}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize student face embeddings.")
    parser.add_argument('--force', action='store_true', help='Regenerate embeddings even if they exist.')
    args = parser.parse_args()
    main(force=args.force)





# import os
# import sys
# import cv2
# import numpy as np
# import pickle
# import argparse
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
#
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
# from attendance.models import Student
#
# # Argument parser
# parser = argparse.ArgumentParser(description="Generate facial embeddings for students.")
# parser.add_argument('--force', action='store_true', help="Force overwrite existing .npy embedding files.")
# args = parser.parse_args()
#
# # Initialize Face Analyzer
# app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
#
# # Paths
# IMG_FOLDER = os.path.join(settings.BASE_DIR, "media/student_faces")
# EMBEDDING_DIR = os.path.join(settings.BASE_DIR, "media/embeddings")
# PKL_PATH = os.path.join(settings.BASE_DIR, "media/face_embeddings.pkl")
# os.makedirs(EMBEDDING_DIR, exist_ok=True)
#
# # Result dictionary
# embeddings_dict = {}
#
# # Process each student
# for student in Student.objects.all():
#     npy_path = os.path.join(EMBEDDING_DIR, f"{student.h_code}.npy")
#
#     if not args.force and os.path.exists(npy_path):
#         print(f"⏩ Skipping {student.h_code}: .npy already exists.")
#         continue
#
#     face_vectors = []
#     for file in os.listdir(IMG_FOLDER):
#         if student.h_code.lower() in file.lower():
#             path = os.path.join(IMG_FOLDER, file)
#             img = cv2.imread(path)
#             if img is None:
#                 continue
#             faces = app.get(img)
#             if faces:
#                 face_vectors.append(faces[0].embedding)
#
#     print(f"{student.h_code}: {len(face_vectors)} images used")
#
#     if face_vectors:
#         avg_embedding = np.mean(face_vectors, axis=0)
#         embeddings_dict[student.h_code] = avg_embedding
#         np.save(npy_path, avg_embedding)
#         print(f"✅ Saved .npy for {student.h_code}")
#     else:
#         print(f"⚠️ No valid faces found for {student.h_code}. Skipped.")
#
# # Save fallback .pkl
# with open(PKL_PATH, 'wb') as f:
#     pickle.dump(embeddings_dict, f)
# print(f"✅ Saved fallback embeddings to: {PKL_PATH}")
#
#



# # extras/initialize_embeddings.py
# import os
# import sys
# import cv2
# import numpy as np
# import pickle
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
# from attendance.models import Student
#
# app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
#
# IMG_FOLDER = os.path.join(settings.BASE_DIR, "media/student_faces")
# EMBEDDING_DIR = os.path.join(settings.BASE_DIR, "media/embeddings")
# PKL_PATH = os.path.join(settings.BASE_DIR, "media/face_embeddings.pkl")
#
# os.makedirs(EMBEDDING_DIR, exist_ok=True)
# embeddings_dict = {}
#
# for student in Student.objects.all():
#     face_vectors = []
#     for file in os.listdir(IMG_FOLDER):
#         if student.h_code.lower() in file.lower():
#             path = os.path.join(IMG_FOLDER, file)
#             img = cv2.imread(path)
#             if img is None:
#                 continue
#             faces = app.get(img)
#             if faces:
#                 face_vectors.append(faces[0].embedding)
#
#     print(f"{student.h_code}: {len(face_vectors)} images used")
#
#     if face_vectors:
#         avg_embedding = np.mean(face_vectors, axis=0)
#         embeddings_dict[student.h_code] = avg_embedding
#
#         # Save .npy file
#         np.save(os.path.join(EMBEDDING_DIR, f"{student.h_code}.npy"), avg_embedding)
#
# # Optional: save a combined .pkl file
# with open(PKL_PATH, 'wb') as f:
#     pickle.dump(embeddings_dict, f)
#
# print(f"✅ Saved {len(embeddings_dict)} student embeddings.")
#
#



# # initialize_embeddings.py
# import os
# import sys
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from django.conf import settings
# from attendance.models import Student
# import pickle
# import django
#
#
#
# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# # django.setup()
#
# app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(640, 640))
#
# IMG_FOLDER = os.path.join(settings.BASE_DIR, "media/student_faces")
# EMBEDDING_DIR = os.path.join(settings.BASE_DIR, "media/embeddings")
# PKL_PATH = os.path.join(settings.BASE_DIR, "media/face_embeddings.pkl")
#
# os.makedirs(EMBEDDING_DIR, exist_ok=True)
# embeddings_dict = {}
#
# for student in Student.objects.all():
#     face_vectors = []
#     for file in os.listdir(IMG_FOLDER):
#         if student.h_code.lower() in file.lower():
#             path = os.path.join(IMG_FOLDER, file)
#             img = cv2.imread(path)
#             if img is None:
#                 continue
#             faces = app.get(img)
#             if faces:
#                 face_vectors.append(faces[0].embedding)
#
#     print(f"{student.h_code}: {len(face_vectors)} images used")
#
#     if face_vectors:
#         avg_embedding = np.mean(face_vectors, axis=0)
#         embeddings_dict[student.h_code] = avg_embedding
#
#         # Save individual .npy for real-time recognition
#         np.save(os.path.join(EMBEDDING_DIR, f"{student.h_code}.npy"), avg_embedding)
#
# # Save full pkl file (optional)
# with open(PKL_PATH, 'wb') as f:
#     pickle.dump(embeddings_dict, f)
#
# print(f"✅ Saved {len(embeddings_dict)} student embeddings.")






# # initialize_embeddings.py
# import os
# import sys
#
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from django.conf import settings
# from attendance.models import Student
# import pickle
#
# app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# # app.prepare(ctx_id=0, det_size=(800, 800))  # ← Now matching real-time recognition
# app.prepare(ctx_id=0, det_size=(640, 640))
#
# IMG_FOLDER = os.path.join(settings.BASE_DIR, "media/student_faces")
# EMBEDDING_FILE = os.path.join(settings.BASE_DIR, "media/face_embeddings.pkl")
#
# embeddings = {}
#
# for student in Student.objects.all():
#     face_vectors = []
#     for file in os.listdir(IMG_FOLDER):
#         if student.h_code.lower() in file.lower():
#             path = os.path.join(IMG_FOLDER, file)
#             img = cv2.imread(path)
#             if img is None:
#                 continue
#             faces = app.get(img)
#             if faces:
#                 face_vectors.append(faces[0].embedding)
#
#     print(f"{student.h_code}: {len(face_vectors)} images used")
#
#     if face_vectors:
#         embeddings[student.h_code] = np.mean(face_vectors, axis=0)
#
# # Save
# with open(EMBEDDING_FILE, 'wb') as f:
#     pickle.dump(embeddings, f)
#
# print(f"Saved {len(embeddings)} student embeddings.")





# import os
# import django
#
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# django.setup()
# #  below cmd Should roughly match the number of FaceImage records you have.
# # ls media/embeddings | wc -l
# # import os
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from django.conf import settings
# from attendance.models import Student
# import pickle
#
# app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# # app.prepare(ctx_id=0, det_size=(640, 640))
# app.prepare(ctx_id=0, det_size=(800, 800))
#
# IMG_FOLDER = os.path.join(settings.BASE_DIR, "media/student_faces")
# EMBEDDING_FILE = os.path.join(settings.BASE_DIR, "media/face_embeddings.pkl")

# import os
# import django
#
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# django.setup()
#
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from django.conf import settings
# from attendance.models import Student
# import pickle
