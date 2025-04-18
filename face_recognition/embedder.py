# face_recognition/embedder.py
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0)

def get_face_embeddings(image):
    faces = app.get(image)
    return faces  # each face has `.embedding`, `.bbox`, etc.
