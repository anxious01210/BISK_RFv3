#  below cmd Should roughly match the number of FaceImage records you have.
# ls media/embeddings | wc -l
import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from django.conf import settings
from attendance.models import Student
import pickle

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

IMG_FOLDER = os.path.join(settings.BASE_DIR, "media/student_faces")
EMBEDDING_FILE = os.path.join(settings.BASE_DIR, "media/face_embeddings.pkl")

embeddings = {}

for student in Student.objects.all():
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

    if face_vectors:
        embeddings[student.h_code] = np.mean(face_vectors, axis=0)

# Save
with open(EMBEDDING_FILE, 'wb') as f:
    pickle.dump(embeddings, f)

print(f"Saved {len(embeddings)} student embeddings.")
