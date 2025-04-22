# # face_recognition/embedder.py
# from insightface.app import FaceAnalysis
#
# app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
# app.prepare(ctx_id=0)
#
# def get_face_embeddings(image):
#     faces = app.get(image)
#     return faces  # each face has `.embedding`, `.bbox`, etc.
#
# import numpy as np
# from attendance.models import FaceImage, Student
#
# def load_embeddings():
#     """
#     Loads all face embeddings from FaceImage entries in the database.
#     Returns:
#         - embeddings: np.ndarray of shape (N, D)
#         - students: list of corresponding Student objects
#     """
#     embeddings = []
#     students = []
#
#     for face in FaceImage.objects.select_related("student").all():
#         try:
#             emb = np.load(face.embedding_path)
#             embeddings.append(emb)
#             students.append(face.student)
#         except Exception as e:
#             print(f"❌ Failed to load embedding for {face.image_path}: {e}")
#
#     return np.vstack(embeddings), students if embeddings else (np.array([]), [])

# face_recognition/embedder.py
import os
import numpy as np
from insightface.app import FaceAnalysis
from django.conf import settings
from attendance.models import FaceImage, Student

# Initialize the face analysis model
app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0)

def get_face_embeddings(image):
    faces = app.get(image)
    return faces  # each face has `.embedding`, `.bbox`, etc.

def load_embeddings():
    """
    Loads all face embeddings from FaceImage entries in the database.
    Returns:
        - embeddings: np.ndarray of shape (N, D)
        - students: list of corresponding Student objects
    """
    embeddings = []
    students = []

    for face in FaceImage.objects.select_related("student").all():
        # Use absolute path based on MEDIA_ROOT
        full_path = os.path.join(settings.MEDIA_ROOT, face.embedding_path)
        try:
            emb = np.load(full_path)
            embeddings.append(emb)
            students.append(face.student)
        except Exception as e:
            print(f"❌ Failed to load embedding for {face.image_path}: {e}")

    if embeddings:
        return np.vstack(embeddings), students
    else:
        return np.array([]), []
