import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
img_path = os.path.join(BASE_DIR, "media", "student_faces", "H123456_9.jpg")
embedding_path = os.path.join(BASE_DIR, "media", "embeddings", "H123456.npy")  # Adjust if stored elsewhere

# --- Load image ---
img = cv2.imread(img_path)
if img is None:
    print("❌ Failed to load image. Check path or file integrity.")
    exit()

# --- Load reference embedding ---
if not os.path.exists(embedding_path):
    print("❌ Reference embedding file not found:", embedding_path)
    exit()

reference_embedding = np.load(embedding_path).astype(np.float32).reshape(1, -1)

# --- Initialize Face Recognizer ---
recognizer = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
recognizer.prepare(ctx_id=0)

# --- Detect faces and compare embeddings ---
faces = recognizer.get(img)
print(f"✅ Detected {len(faces)} face(s).")

if faces:
    for i, face in enumerate(faces):
        detected_embedding = face.embedding.reshape(1, -1)
        score = cosine_similarity(detected_embedding, reference_embedding)[0][0]
        print(f"👉 Face {i+1}:")
        print(f"    - Embedding shape: {detected_embedding.shape}")
        print(f"    - Match Score (cosine similarity): {score:.4f}")
else:
    print("⚠️ No faces found to compare.")




# import os
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
#
# # --- Load image ---
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# img_path = os.path.join(BASE_DIR, "media", "student_faces", "H123456_9.jpg")
# img = cv2.imread(img_path)
#
# if img is None:
#     print("❌ Failed to load image. Check path or file integrity.")
#     exit()
#
# # --- Initialize Face Recognizer ---
# recognizer = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
# recognizer.prepare(ctx_id=0)
#
# # --- Detect and extract embeddings ---
# faces = recognizer.get(img)
# print(f"Detected {len(faces)} face(s).")
#
# # --- Example: Reference embedding (e.g., stored one) ---
# # In practice, load this from database or file (here it's hardcoded for demonstration)
# # Dummy example: reference_embedding = np.load("embedding_H123456.npy")
# # For now, simulate with a random vector of same shape (512-d by default)
# reference_embedding = np.random.rand(512).astype(np.float32)
#
# if faces:
#     for i, face in enumerate(faces):
#         embedding = face.embedding.reshape(1, -1)
#         reference = reference_embedding.reshape(1, -1)
#         score = cosine_similarity(embedding, reference)[0][0]
#         print(f"Face {i+1}: Embedding shape: {embedding.shape}, Match Score (cosine similarity): {score:.4f}")
#
#


# import os
# import cv2
# from insightface.app import FaceAnalysis
#
# # img = cv2.imread("media/student_faces/H123456.png")  # Replace with one of your known images
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# # img_path = os.path.join(BASE_DIR, "media", "student_faces", "H123456.png")
# img_path = os.path.join(BASE_DIR, "media", "student_faces", "H123456_9.jpg")
# img = cv2.imread(img_path)
#
# if img is None:
#     print("❌ Failed to load image. Check path or file integrity.")
#     exit()
#
# recognizer = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
# recognizer.prepare(ctx_id=0)
#
# faces = recognizer.get(img)
# print(f"Detected {len(faces)} faces.")
# if faces:
#     for face in faces:
#         print("Embedding shape:", face.embedding.shape)
