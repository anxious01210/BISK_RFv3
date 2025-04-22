import cv2
import os
import numpy as np
from datetime import datetime
from django.utils import timezone
import django
import sys

# Setup Django environment
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
django.setup()

from attendance.models import Camera, Period, AttendanceLog, Student, FaceImage, RecognitionSchedule
from face_recognition.embedder import load_embeddings, get_face_embeddings
from retinaface import RetinaFace
from insightface.app import FaceAnalysis

# Constants
THRESHOLD = 0.85  # Cosine similarity threshold

# Load detector and recognizer
face_detector = RetinaFace(model_name='mobile0.25')
recognizer = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
recognizer.prepare(ctx_id=0)

# Load all embeddings from database
known_embeddings, known_students = load_embeddings()

def is_within_period(current_time, period):
    return period.start_time <= current_time.time() <= period.end_time

def is_within_camera_schedule(current_time, camera):
    return camera.is_active and (not camera.stream_start_time or camera.stream_start_time <= current_time.time()) and (not camera.stream_end_time or current_time.time() <= camera.stream_end_time)

def is_within_recognition_schedule(current_time, schedule):
    if not schedule.is_active:
        return False
    if current_time.weekday() not in schedule.weekday_values:
        return False
    return schedule.start_time <= current_time.time() <= schedule.end_time

def already_marked(student, period):
    return AttendanceLog.objects.filter(student=student, period=period, timestamp__date=timezone.now().date()).exists()

def recognize_and_log():
    current_time = timezone.localtime()
    active_periods = Period.objects.filter(is_active=True)
    active_schedules = RecognitionSchedule.objects.filter(is_active=True)

    for camera in Camera.objects.filter(is_active=True):
        if not is_within_camera_schedule(current_time, camera):
            continue

        matched_schedule = next((s for s in active_schedules if s.camera == camera and is_within_recognition_schedule(current_time, s)), None)
        if not matched_schedule:
            continue

        print(f"📷 Starting stream: {camera.name} at {current_time.strftime('%H:%M:%S')}")
        cap = cv2.VideoCapture(camera.url)

        if not cap.isOpened():
            print(f"❌ Failed to open camera: {camera.name}")
            continue

        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("❌ Failed to capture frame")
            continue

        faces = face_detector.detect_faces(frame)

        for face in faces:
            x1, y1, x2, y2 = face["facial_area"]
            face_crop = frame[y1:y2, x1:x2]
            face_embedding = get_face_embeddings(face_crop)[0]

            # Compare with all known embeddings
            similarities = np.dot(known_embeddings, face_embedding.T)
            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]

            if best_score >= THRESHOLD:
                student = known_students[best_idx]
                print(f"✅ Match: {student.full_name} (Score: {best_score:.4f})")

                for period in active_periods:
                    if is_within_period(current_time, period) and not already_marked(student, period):
                        AttendanceLog.objects.create(
                            student=student,
                            camera=camera,
                            period=period,
                            timestamp=current_time
                        )
                        print(f"📝 Attendance logged for {student.h_code} in period {period.name}")
            else:
                print("❌ Unknown face")


if __name__ == "__main__":
    recognize_and_log()
