# face_utils.py
import os
import cv2
import numpy as np
import pickle
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
# from moviepy.editor import VideoFileClip  # Uncomment only if needed

# Constants
EMBEDDING_DIR = 'media/embeddings'
PKL_PATH = 'media/face_embeddings.pkl'
PROCESSED_DIR = 'media/face_inspector_processed'
THRESHOLD = 0.3

# Lazy model initialization to avoid multiple model loading
def get_face_analyzer():
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0)
    return app

# Load all .npy reference embeddings
def load_reference_embeddings():
    embeddings = {}
    for file in os.listdir(EMBEDDING_DIR):
        if file.endswith('.npy'):
            path = os.path.join(EMBEDDING_DIR, file)
            try:
                h_code = file.replace('.npy', '')
                embeddings[h_code] = np.load(path).astype(np.float32)
            except Exception as e:
                print(f"⚠️ Error loading {file}: {e}")
    return embeddings

# Load backup .pkl embeddings
def load_backup_embeddings():
    if os.path.exists(PKL_PATH):
        try:
            with open(PKL_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load PKL: {e}")
    return {}

# Matching logic
def match_face(embedding, ref_embeds, backup_embeds):
    best_score = -1
    best_id = None
    for h_code, ref in ref_embeds.items():
        score = cosine_similarity(embedding.reshape(1, -1), ref.reshape(1, -1))[0][0]
        if score > best_score:
            best_score = score
            best_id = h_code

    if best_score >= THRESHOLD:
        return best_id, best_score

    for h_code, data in backup_embeds.items():
        if isinstance(data, dict) and 'embedding' in data:
            ref = data['embedding']
            score = cosine_similarity(embedding.reshape(1, -1), ref.reshape(1, -1))[0][0]
            if score > best_score:
                best_score = score
                best_id = h_code

    return (best_id, best_score) if best_score >= THRESHOLD else (None, 0.0)

# Extract best face from video
def extract_best_face_from_video(video_path, app):
    cap = cv2.VideoCapture(video_path)
    best_face = None
    best_embedding = None
    max_area = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                best_face = frame[y1:y2, x1:x2]
                best_embedding = face.embedding
                max_area = area
    cap.release()
    return best_face, best_embedding

# Main processing function
def process_uploaded_media(file_path, media_type):
    app = get_face_analyzer()
    ref_embeds = load_reference_embeddings()
    backup_embeds = load_backup_embeddings()

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(PROCESSED_DIR, base_name)
    os.makedirs(output_dir, exist_ok=True)

    if media_type == 'video':
        face_img, emb = extract_best_face_from_video(file_path, app)
        if face_img is not None:
            h_code, score = match_face(emb, ref_embeds, backup_embeds)
            name = backup_embeds.get(h_code, {}).get('name', 'Unknown') if h_code else 'Unknown'
            # save_path = os.path.join(output_dir, f"{h_code or 'unknown'}__{name}.jpg")
            # save_path = os.path.join(output_dir, f"{h_code or 'unknown'}__{name}_{idx + 1}_score={score:.2f}.jpg")
            save_path = os.path.join(output_dir, f"{h_code or 'unknown'}__{name}_score={score:.2f}.jpg")
            cv2.imwrite(save_path, face_img)
            print(f"📽️ Processed video: saved {save_path}")
        else:
            print(f"⚠️ No face found in video: {file_path}")
    else:
        img = cv2.imread(file_path)
        if img is None:
            print(f"❌ Could not read image: {file_path}")
            return

        faces = app.get(img)
        print(f"🧠 Detected {len(faces)} face(s) in {file_path}")

        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face.bbox.astype(int)
            cropped = img[y1:y2, x1:x2]
            h_code, score = match_face(face.embedding, ref_embeds, backup_embeds)

            if h_code:
                name = backup_embeds.get(h_code, {}).get('name', 'Unknown')
                print(f"✅ Match: h_code='{h_code}' name='{name}' score={score:.3f}")
            else:
                name = "Unknown"
                print(f"⚠️ No match found. score={score:.3f}")

            # save_path = os.path.join(output_dir, f"{h_code or 'unknown'}__{name}_{idx+1}.jpg")
            save_path = os.path.join(output_dir, f"{h_code or 'unknown'}__{name}_{idx + 1}_score={score:.2f}.jpg")
            cv2.imwrite(save_path, cropped)
            print(f"📸 Saved: {save_path}")




# import sys
# print(">>> sys.path:", sys.path)
# # face_utils.py
# import os
# import cv2
# import numpy as np
# import pickle
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from moviepy import VideoFileClip  # You confirmed this works in your setup
#
# # Constants
# EMBEDDING_DIR = 'media/embeddings'
# PKL_PATH = 'media/face_embeddings.pkl'
# PROCESSED_DIR = 'media/face_inspector_processed'
# THRESHOLD = 0.3
#
# # Initialize face analyzer
# app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
# app.prepare(ctx_id=0)
#
# # Load all .npy reference embeddings
# def load_reference_embeddings():
#     embeddings = {}
#     for file in os.listdir(EMBEDDING_DIR):
#         if file.endswith('.npy'):
#             h_code = file.replace('.npy', '')
#             embeddings[h_code] = np.load(os.path.join(EMBEDDING_DIR, file)).astype(np.float32)
#     return embeddings
#
# # Load backup .pkl embeddings
# def load_backup_embeddings():
#     if os.path.exists(PKL_PATH):
#         with open(PKL_PATH, 'rb') as f:
#             return pickle.load(f)
#     return {}
#
# # Matching logic
# def match_face(embedding, ref_embeds, backup_embeds):
#     best_score = -1
#     best_id = None
#     for h_code, ref in ref_embeds.items():
#         score = cosine_similarity(embedding.reshape(1, -1), ref.reshape(1, -1))[0][0]
#         if score > best_score:
#             best_score = score
#             best_id = h_code
#
#     if best_score >= THRESHOLD:
#         return best_id, best_score
#
#     # Try backup if nothing found
#     for h_code, data in backup_embeds.items():
#         if isinstance(data, dict) and 'embedding' in data:
#             ref = data['embedding']
#             score = cosine_similarity(embedding.reshape(1, -1), ref.reshape(1, -1))[0][0]
#             if score > best_score:
#                 best_score = score
#                 best_id = h_code
#
#     if best_score >= THRESHOLD:
#         return best_id, best_score
#     return None, 0.0
#
# # Extract best face from video
# def extract_best_face_from_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     best_face = None
#     best_embedding = None
#     max_area = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         faces = app.get(frame)
#         for face in faces:
#             x1, y1, x2, y2 = face.bbox.astype(int)
#             area = (x2 - x1) * (y2 - y1)
#             if area > max_area:
#                 best_face = frame[y1:y2, x1:x2]
#                 best_embedding = face.embedding
#                 max_area = area
#     cap.release()
#     return best_face, best_embedding
#
# # Main processing function
# def process_uploaded_media(file_path, media_type):
#     ref_embeds = load_reference_embeddings()
#     backup_embeds = load_backup_embeddings()
#
#     base_name = os.path.basename(file_path).split('.')[0]
#     output_dir = os.path.join(PROCESSED_DIR, base_name)
#     os.makedirs(output_dir, exist_ok=True)
#
#     if media_type == 'video':
#         face_img, emb = extract_best_face_from_video(file_path)
#         if face_img is not None:
#             h_code, _ = match_face(emb, ref_embeds, backup_embeds)
#             data = backup_embeds.get(h_code, None)
#             name = data.get('name', 'Unknown') if isinstance(data, dict) else 'Unknown'
#             save_path = os.path.join(output_dir, f"{h_code or 'unknown'}__{name}.jpg")
#             cv2.imwrite(save_path, face_img)
#
#     else:
#         img = cv2.imread(file_path)
#         if img is None:
#             print(f"❌ Could not read image: {file_path}")
#             return
#
#         faces = app.get(img)
#         print(f"🧠 Detected {len(faces)} face(s) in {file_path}")
#
#         for idx, face in enumerate(faces):
#             x1, y1, x2, y2 = face.bbox.astype(int)
#             cropped = img[y1:y2, x1:x2]
#             h_code, score = match_face(face.embedding, ref_embeds, backup_embeds)
#
#             if h_code:
#                 name = backup_embeds.get(h_code, {}).get('name', 'Unknown')
#                 print(f"✅ Match: {h_code=} {name=} {score=:.3f}")
#             else:
#                 name = "Unknown"
#                 print(f"⚠️ No match found. {score=:.3f}")
#
#             save_path = os.path.join(output_dir, f"{h_code or 'unknown'}__{name}_{idx+1}.jpg")
#             cv2.imwrite(save_path, cropped)
#             print(f"📸 Saved: {save_path}")
#

    # else:  # image
    #     img = cv2.imread(file_path)
    #     if img is None:
    #         print(f"❌ Failed to read image: {file_path}")
    #         return
    #     faces = app.get(img)
    #     for idx, face in enumerate(faces):
    #         x1, y1, x2, y2 = face.bbox.astype(int)
    #         cropped = img[y1:y2, x1:x2]
    #         h_code, _ = match_face(face.embedding, ref_embeds, backup_embeds)
    #         data = backup_embeds.get(h_code, None)
    #         name = data.get('name', 'Unknown') if isinstance(data, dict) else 'Unknown'
    #         save_path = os.path.join(output_dir, f"{h_code or 'unknown'}__{name}_{idx+1}.jpg")
    #         cv2.imwrite(save_path, cropped)








# import sys
# print(">>> sys.path:", sys.path)
#
# import os
# import cv2
# import numpy as np
# import pickle
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# # from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
# from moviepy import VideoFileClip
#
#
# EMBEDDING_DIR = 'media/embeddings'
# PKL_PATH = 'media/face_embeddings.pkl'
# PROCESSED_DIR = 'media/face_inspector_processed'
# THRESHOLD = 0.5
#
# app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
# app.prepare(ctx_id=0)
#
# # Load all .npy reference embeddings
# def load_reference_embeddings():
#     embeddings = {}
#     for file in os.listdir(EMBEDDING_DIR):
#         if file.endswith('.npy'):
#             h_code = file.replace('.npy', '')
#             embeddings[h_code] = np.load(os.path.join(EMBEDDING_DIR, file)).astype(np.float32)
#     return embeddings
#
# # Load .pkl fallback
# def load_backup_embeddings():
#     if os.path.exists(PKL_PATH):
#         with open(PKL_PATH, 'rb') as f:
#             return pickle.load(f)
#     return {}
#
# def match_face(embedding, ref_embeds, backup_embeds):
#     best_score = -1
#     best_id = None
#     for h_code, ref in ref_embeds.items():
#         score = cosine_similarity(embedding.reshape(1, -1), ref.reshape(1, -1))[0][0]
#         if score > best_score:
#             best_score = score
#             best_id = h_code
#
#     if best_score >= THRESHOLD:
#         return best_id, best_score
#
#     # Try backup if nothing found
#     for h_code, data in backup_embeds.items():
#         ref = data['embedding']
#         score = cosine_similarity(embedding.reshape(1, -1), ref.reshape(1, -1))[0][0]
#         if score > best_score:
#             best_score = score
#             best_id = h_code
#
#     if best_score >= THRESHOLD:
#         return best_id, best_score
#     return None, 0.0
#
# def extract_best_face_from_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     best_face = None
#     best_embedding = None
#     max_area = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         faces = app.get(frame)
#         for face in faces:
#             x1, y1, x2, y2 = face.bbox.astype(int)
#             area = (x2 - x1) * (y2 - y1)
#             if area > max_area:
#                 best_face = frame[y1:y2, x1:x2]
#                 best_embedding = face.embedding
#                 max_area = area
#     cap.release()
#     return best_face, best_embedding
#
# def process_uploaded_media(file_path, media_type):
#     ref_embeds = load_reference_embeddings()
#     backup_embeds = load_backup_embeddings()
#
#     base_name = os.path.basename(file_path).split('.')[0]
#     output_dir = os.path.join(PROCESSED_DIR, base_name)
#     os.makedirs(output_dir, exist_ok=True)
#
#     if media_type == 'video':
#         face_img, emb = extract_best_face_from_video(file_path)
#         if face_img is not None:
#             h_code, _ = match_face(emb, ref_embeds, backup_embeds)
#             name = backup_embeds.get(h_code, {}).get('name', 'Unknown')
#             save_path = os.path.join(output_dir, f"{h_code or 'unknown'}__{name}.jpg")
#             cv2.imwrite(save_path, face_img)
#     else:
#         img = cv2.imread(file_path)
#         faces = app.get(img)
#         for idx, face in enumerate(faces):
#             x1, y1, x2, y2 = face.bbox.astype(int)
#             cropped = img[y1:y2, x1:x2]
#             h_code, _ = match_face(face.embedding, ref_embeds, backup_embeds)
#             name = backup_embeds.get(h_code, {}).get('name', 'Unknown')
#             save_path = os.path.join(output_dir, f"{h_code or 'unknown'}__{name}_{idx+1}.jpg")
#             cv2.imwrite(save_path, cropped)
