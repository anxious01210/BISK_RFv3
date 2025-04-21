import os
import django
import cv2
from insightface.app import FaceAnalysis
from numpy.linalg import norm
import numpy as np
from django.conf import settings

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
django.setup()

from attendance.models import Student, FaceImage

# InsightFace App
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Path where images are stored
FACE_IMAGE_DIR = os.path.join(settings.MEDIA_ROOT, 'student_faces')
EMBEDDINGS_DIR = os.path.join(settings.MEDIA_ROOT, 'embeddings')
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Clear old embeddings
FaceImage.objects.all().delete()

# Process images
for file_name in os.listdir(FACE_IMAGE_DIR):
    file_path = os.path.join(FACE_IMAGE_DIR, file_name)
    if not os.path.isfile(file_path):
        continue

    # Try to extract h_code from filename
    matched_student = None
    for student in Student.objects.all():
        if student.h_code.lower() in file_name.lower():
            matched_student = student
            break

    if not matched_student:
        print(f"❌ No student found for file: {file_name}")
        continue

    img = cv2.imread(file_path)
    if img is None:
        print(f"⚠️ Could not read image: {file_name}")
        continue

    faces = app.get(img)
    if not faces:
        print(f"🚫 No face detected in: {file_name}")
        continue

    # Get first face's embedding
    face = faces[0]
    embedding = face.embedding
    emb_path = os.path.join('embeddings', file_name + '.npy')
    full_emb_path = os.path.join(settings.MEDIA_ROOT, emb_path)
    np.save(full_emb_path, embedding)

    # Save to DB
    FaceImage.objects.create(
        student=matched_student,
        image_path=os.path.join('student_faces', file_name),
        embedding_path=emb_path
    )

    print(f"✅ Processed: {file_name} → Student: {matched_student.h_code}")

print("\n🎉 Embedding initialization complete!")
