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

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
import django
django.setup()

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from django.conf import settings
from attendance.models import Student
import pickle

app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# app.prepare(ctx_id=0, det_size=(800, 800))  # ← Now matching real-time recognition
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

    print(f"{student.h_code}: {len(face_vectors)} images used")

    if face_vectors:
        embeddings[student.h_code] = np.mean(face_vectors, axis=0)

# Save
with open(EMBEDDING_FILE, 'wb') as f:
    pickle.dump(embeddings, f)

print(f"Saved {len(embeddings)} student embeddings.")
