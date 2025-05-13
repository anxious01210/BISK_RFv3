# This is the same as the already created utils_opencv.py
# Reposting here in a new canvas as requested

import os
import cv2
import time
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from django.core.files.base import ContentFile
from django.conf import settings

from attendance.models import AttendanceLog, Student, Period
from extras.log_utils import get_camera_logger

MATCH_THRESHOLD = 0.60
RESTART_LIMIT = 5
RESTART_DELAY = 10  # seconds

LOG_ATTENDANCE = getattr(settings, "LOG_ATTENDANCE", True)
LOG_MATCHES = getattr(settings, "LOG_MATCHES", True)
LOG_CROPPED_DEBUG = getattr(settings, "LOG_CROPPED_DEBUG", True)
LOG_ATTENDANCE_UPDATED = getattr(settings, "LOG_ATTENDANCE_UPDATED", True)
LOG_MATCH_THRESHOLD = getattr(settings, "LOG_MATCH_THRESHOLD", True)

def get_attendance_crop_path(student_h_code, camera_name):
    today = datetime.now()
    return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"

def load_embeddings(embedding_dir):
    import pickle
    embeddings_map = {}
    npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    if npy_files:
        for file in npy_files:
            h_code = file.split('.')[0]
            embedding = np.load(os.path.join(embedding_dir, file))
            embeddings_map[h_code] = embedding
        print(f"✅ Loaded {len(embeddings_map)} face embeddings")
    else:
        fallback = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
        if os.path.exists(fallback):
            with open(fallback, "rb") as f:
                embeddings_map = pickle.load(f)
            print("⚠️ Loaded fallback face_embeddings.pkl")
        else:
            print("❌ No embeddings found")
    return embeddings_map

def process_camera_stream_opencv(camera, schedules, face_analyzer, embeddings_map):
    from django.db import connections
    connections.close_all()

    logger = get_camera_logger(camera.name)
    logger.info(f"🟢 OpenCV stream started for camera: {camera.name}")
    logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
    logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
    logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")

    try:
        active_students = Student.objects.filter(is_active=True).count()
        logger.info(f"🧑‍🎓 Active students: {active_students}")
    except Exception as e:
        logger.warning(f"⚠️ Failed to query active students: {e}")

    cap = cv2.VideoCapture(camera.url)
    if not cap.isOpened():
        logger.warning(f"⚠️ Cannot open stream for {camera.name}")
        return

    if LOG_MATCH_THRESHOLD:
        logger.info(f"🔴 MATCH_THRESHOLD = {MATCH_THRESHOLD:.2f}")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            logger.warning(f"⚠️ Frame read failure from {camera.name}")
            break

        current_time = datetime.now()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb)

        for face in faces:
            if face.embedding is None:
                continue

            best_score = -1
            best_h_code = None
            for h_code, known_embedding in embeddings_map.items():
                try:
                    known_norm = known_embedding / np.linalg.norm(known_embedding)
                    face_norm = face.embedding / np.linalg.norm(face.embedding)
                    score = cosine_similarity([face_norm], [known_norm])[0][0]
                    if score > best_score:
                        best_score = score
                        best_h_code = h_code
                except Exception as e:
                    logger.warning(f"Similarity error with {h_code}: {e}")

            if best_score >= MATCH_THRESHOLD and best_h_code:
                try:
                    student = Student.objects.get(h_code=best_h_code, is_active=True)
                    display_name = f"{student.h_code}__{student.full_name}"
                    if LOG_MATCHES:
                        logger.info(f"🧬 Best match {display_name} | Score: {best_score:.4f}")
                except Student.DoesNotExist:
                    logger.warning(f"❌ Student not found for H-code: {best_h_code}")
                    continue

                active_periods = Period.objects.filter(is_active=True)
                for period in active_periods:
                    if period.start_time <= current_time.time() <= period.end_time:
                        log, created = AttendanceLog.objects.get_or_create(
                            student=student,
                            period=period,
                            date=current_time.date(),
                            defaults={
                                "camera": camera,
                                "timestamp": current_time,
                                "match_score": best_score
                            }
                        )

                        updated = False
                        if not created:
                            if best_score > log.match_score:
                                log.timestamp = current_time
                                log.match_score = best_score
                                log.camera = camera
                                updated = True
                                if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
                                    log.cropped_face.delete(save=False)

                        if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
                            try:
                                x1, y1, x2, y2 = map(int, face.bbox)
                                cropped_face = rgb[y1:y2, x1:x2]
                                _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                                log.cropped_face.save(
                                    get_attendance_crop_path(student.h_code, camera.name),
                                    ContentFile(img_encoded.tobytes()),
                                    save=False
                                )
                                updated = True
                            except Exception as e:
                                logger.error(f"❌ Saving cropped image failed: {e}")

                        if created:
                            log.save()
                            if LOG_ATTENDANCE:
                                logger.info(f"✅ Attendance saved for {display_name} in period '{period.name}' (Score: {best_score:.4f})")
                        elif updated:
                            log.save()
                            if LOG_ATTENDANCE_UPDATED:
                                logger.info(f"🔁 Attendance updated for {display_name} in period '{period.name}' (New Score: {best_score:.4f})")
                        else:
                            if LOG_ATTENDANCE_UPDATED:
                                logger.info(f"🟡 Attendance already exists with equal or better score for {display_name} in period '{period.name}'")
            else:
                logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")

            if getattr(settings, "SAVE_ALL_CROPPED_FACES", True) and LOG_CROPPED_DEBUG:
                try:
                    x1, y1, x2, y2 = map(int, face.bbox)
                    cropped_face = rgb[y1:y2, x1:x2]
                    debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
                    os.makedirs(debug_dir, exist_ok=True)
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    debug_path = os.path.join(debug_dir, filename)
                    cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                    logger.info(f"🔄 Debug cropped face saved: {debug_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Debug save failed: {e}")

    cap.release()
    logger.info(f"🚩 Stopped streaming from camera: {camera.name}")
