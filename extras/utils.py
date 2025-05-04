# extras/utils.py  (Good with 10 Sec auto-restart for stream, to be set at the end of the code.)
# Now if you change the camera stream directly from camera, the script after 10 consecutive corupted frames it will
# it automatically restarts the stream to handle the camera resolution changes from the camera.

# extras/utils.py
import cv2
import numpy as np
from django.conf import settings
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import logging
import os
import time
from insightface.utils.face_align import norm_crop
from django.core.files.base import ContentFile
import io

from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
from extras.log_utils import get_camera_logger

MATCH_THRESHOLD = 0.50

RESTART_LIMIT = 5
RESTART_DELAY = 10  # seconds


def is_within_recognition_schedule(current_time, schedule):
    weekday = current_time.strftime('%A')
    return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time


def load_embeddings(embedding_dir):
    import pickle
    embeddings_map = {}
    npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
    if npy_files:
        for file in npy_files:
            h_code = file.split('.')[0]
            embedding = np.load(os.path.join(embedding_dir, file))
            embeddings_map[h_code] = embedding
    else:
        fallback = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
        if os.path.exists(fallback):
            with open(fallback, "rb") as f:
                embeddings_map = pickle.load(f)
            print("⚠️ Loaded fallback face_embeddings.pkl")
        else:
            print("❌ No embeddings found")
    return embeddings_map


def get_attendance_crop_path(student_h_code, camera_name):
    today = datetime.now()
    return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"


def process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts=0):
    import uuid
    logger = get_camera_logger(camera.name)

    logger.info(f"🎥 [START] Processing stream for: {camera.name}")
    logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
    logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
    logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")

    cap = cv2.VideoCapture(camera.url)
    if not cap.isOpened():
        logger.warning(f"⚠️ Cannot open stream for {camera.name}")
        logger.info(f"🔁 Attempting auto-restart for {camera.name} in {RESTART_DELAY} seconds...")
        time.sleep(RESTART_DELAY)
        if restart_attempts < RESTART_LIMIT:
            process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts + 1)
        else:
            logger.error(f"🛑 Max auto-restart attempts reached for {camera.name}. Giving up.")
        return

    logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
    use_alignment = hasattr(face_analyzer, 'align_crop')
    if use_alignment:
        logger.info("✅ Face alignment (align_crop) is ENABLED.")
    else:
        logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")

    read_failure_count = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            read_failure_count += 1
            logger.warning(f"⚠️ Frame read failure ({read_failure_count}) from {camera.name}")
            if read_failure_count >= 30:
                logger.error(f"🔥 Stopping {camera.name} after 30 read failures.")
                break
            time.sleep(10)
            continue
        read_failure_count = 0

        current_time = datetime.now()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb)
        aligned_faces = []

        for face in faces:
            if face.embedding is None or len(face.embedding) == 0:
                continue
            face_box = face.bbox.astype(int)
            x1, y1, x2, y2 = face_box
            cropped_face = rgb[y1:y2, x1:x2]
            if cropped_face is None or cropped_face.size == 0:
                continue

            if use_alignment:
                try:
                    aligned_face = face_analyzer.align_crop(rgb, face.kps)
                    aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
                    face.embedding = aligned_embedding
                except Exception as e:
                    logger.warning(f"❌ Alignment failed: {e}")
                    continue
            aligned_faces.append((face, cropped_face))

        for face, cropped_face in aligned_faces:
            best_score = -1
            best_h_code = None

            for h_code, known_embedding in embeddings_map.items():
                try:
                    score = cosine_similarity([face.embedding], [known_embedding])[0][0]
                    if score > best_score:
                        best_score = score
                        best_h_code = h_code
                except Exception as e:
                    logger.warning(f"Similarity error: {e}")

            if best_score >= MATCH_THRESHOLD and best_h_code:
                try:
                    student = Student.objects.get(h_code=best_h_code, is_active=True)
                except Student.DoesNotExist:
                    logger.warning(f"❌ Student not found for H-code: {best_h_code}")
                    continue

                for sched in schedules:
                    if not is_within_recognition_schedule(current_time, sched):
                        continue
                    period = sched.period

                    log, created = AttendanceLog.objects.get_or_create(
                        student=student,
                        period=period,
                        date=current_time.date(),
                        defaults={"camera": camera, "timestamp": current_time, "match_score": best_score}
                    )
                    if not created:
                        delta = abs((current_time - log.timestamp).total_seconds())
                        if delta <= 3600 and best_score > log.match_score:
                            log.timestamp = current_time
                            log.match_score = best_score
                            log.camera = camera
                            if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
                                log.cropped_face.delete(save=False)
                    if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
                        try:
                            _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                            log.cropped_face.save(
                                get_attendance_crop_path(student.h_code, camera.name),
                                ContentFile(img_encoded.tobytes()),
                                save=False
                            )
                        except Exception as e:
                            logger.error(f"❌ Saving cropped image failed: {e}")
                    log.save()
                    logger.info(f"✅ Attendance saved for {student.full_name} (Score: {best_score:.4f})")
            else:
                logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")

            if getattr(settings, "SAVE_ALL_CROPPED_FACES", True):
                try:
                    debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
                    os.makedirs(debug_dir, exist_ok=True)
                    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
                    debug_path = os.path.join(debug_dir, filename)
                    cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                    logger.info(f"🧪 Debug cropped face saved: {debug_path}")
                except Exception as e:
                    logger.warning(f"⚠️ Debug save failed: {e}")

    cap.release()
    logger.info(f"🚩 Stopped streaming from camera: {camera.name}")
    logger.info(f"🔁 Attempting auto-restart for {camera.name} in {RESTART_DELAY} seconds...")
    time.sleep(RESTART_DELAY)
    if restart_attempts < RESTART_LIMIT:
        process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts + 1)
    else:
        logger.error(f"🛑 Max auto-restart attempts reached for {camera.name}. Giving up.")















#
# # extras/utils.py
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.20
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     import pickle
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding = np.load(os.path.join(embedding_dir, file))
#             embeddings_map[h_code] = embedding
#     else:
#         fallback = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(fallback):
#             with open(fallback, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found")
#     return embeddings_map
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts=0):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"🎥 [START] Processing stream for: {camera.name}")
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Cannot open stream for {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None or frame.size == 0:
#             read_failure_count += 1
#             logger.warning(f"⚠️ Frame read failure ({read_failure_count}) from {camera.name}")
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Stopping {camera.name} after 30 read failures.")
#                 break
#             time.sleep(10)
#             continue
#         read_failure_count = 0
#
#         current_time = datetime.now()
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#                 continue
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#             if cropped_face is None or cropped_face.size == 0:
#                 continue
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment failed: {e}")
#                     continue
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Similarity error: {e}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ Student not found for H-code: {best_h_code}")
#                     continue
#
#                 for sched in schedules:
#                     if not is_within_recognition_schedule(current_time, sched):
#                         continue
#                     period = sched.period
#
#                     log, created = AttendanceLog.objects.get_or_create(
#                         student=student,
#                         period=period,
#                         date=current_time.date(),
#                         defaults={"camera": camera, "timestamp": current_time, "match_score": best_score}
#                     )
#                     if not created:
#                         delta = abs((current_time - log.timestamp).total_seconds())
#                         if delta <= 3600 and best_score > log.match_score:
#                             log.timestamp = current_time
#                             log.match_score = best_score
#                             log.camera = camera
#                             if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                                 log.cropped_face.delete(save=False)
#                     if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                         try:
#                             _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                             log.cropped_face.save(
#                                 get_attendance_crop_path(student.h_code, camera.name),
#                                 ContentFile(img_encoded.tobytes()),
#                                 save=False
#                             )
#                         except Exception as e:
#                             logger.error(f"❌ Saving cropped image failed: {e}")
#                     log.save()
#                     logger.info(f"✅ Attendance saved for {student.full_name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#             if getattr(settings, "SAVE_ALL_CROPPED_FACES", True):
#                 try:
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")
#
#     cap.release()
#     logger.info(f"🚩 Stopped streaming from camera: {camera.name}")
#     logger.info(f"🔁 Attempting auto-restart for {camera.name} in 10 seconds...")
#     time.sleep(10)
#     if restart_attempts < 5:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts + 1)
#     else:
#         logger.error(f"🛑 Max auto-restart attempts reached for {camera.name}. Giving up.")
#























# # extras/utils.py
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.50  # Lowered for better sensitivity
#
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
#
# def load_embeddings(embedding_dir):
#     import pickle
#
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     else:
#         pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(pkl_path):
#             with open(pkl_path, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found: .npy files and .pkl are both missing")
#
#     return embeddings_map
#
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts=0):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None or frame.size == 0:
#             logger.warning(f"⚠️ Invalid frame from {camera.name}")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(10)
#             continue
#         read_failure_count = 0
#
#         current_time = datetime.now()
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#                 continue
#
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#             if cropped_face is None or cropped_face.size == 0:
#                 continue
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment failed: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Error comparing embeddings: {e}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                 except Student.DoesNotExist:
#                     logger.warning(f"No student for H-code: {best_h_code}")
#                     continue
#
#                 for sched in schedules:
#                     if not is_within_recognition_schedule(current_time, sched):
#                         continue
#                     period = sched.period
#
#                     log, created = AttendanceLog.objects.get_or_create(
#                         student=student,
#                         period=period,
#                         date=current_time.date(),
#                         defaults={"camera": camera, "timestamp": current_time, "match_score": best_score}
#                     )
#
#                     if not created:
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#                         if time_diff <= 3600 and best_score > log.match_score:
#                             log.timestamp = current_time
#                             log.match_score = best_score
#                             log.camera = camera
#
#                             if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                                 log.cropped_face.delete(save=False)
#
#                     if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                         try:
#                             _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                             log.cropped_face.save(
#                                 get_attendance_crop_path(student.h_code, camera.name),
#                                 ContentFile(img_encoded.tobytes()),
#                                 save=False
#                             )
#                         except Exception as e:
#                             logger.error(f"Image save error: {e}")
#
#                     log.save()
#                     logger.info(f"✅ Attendance logged for {student.full_name} (Score: {best_score:.4f})")
#
#             if getattr(settings, 'SAVE_ALL_CROPPED_FACES', False):
#                 try:
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                 except Exception as e:
#                     logger.error(f"Error saving debug cropped face: {e}")
#
#     cap.release()
#     logger.info(f"🚩 Stopped streaming from camera: {camera.name}")
#     logger.info(f"🔁 Attempting auto-restart for {camera.name} in 10 seconds...")
#     time.sleep(10)
#     if restart_attempts < 5:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts + 1)
#     else:
#         logger.error(f"🛑 Max auto-restart attempts reached for {camera.name}. Giving up.")




# # extras/utils.py
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.50  # Lowered for better sensitivity
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     import pickle
#
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     else:
#         pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(pkl_path):
#             with open(pkl_path, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found: .npy files and .pkl are both missing")
#
#     return embeddings_map
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None or frame.size == 0:
#             logger.warning(f"⚠️ Invalid frame from {camera.name}")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(10)
#             continue
#         read_failure_count = 0
#
#         current_time = datetime.now()
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#                 continue
#
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#             if cropped_face is None or cropped_face.size == 0:
#                 continue
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment failed: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Error comparing embeddings: {e}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                 except Student.DoesNotExist:
#                     logger.warning(f"No student for H-code: {best_h_code}")
#                     continue
#
#                 for sched in schedules:
#                     if not is_within_recognition_schedule(current_time, sched):
#                         continue
#                     period = sched.period
#
#                     log, created = AttendanceLog.objects.get_or_create(
#                         student=student,
#                         period=period,
#                         date=current_time.date(),
#                         defaults={"camera": camera, "timestamp": current_time, "match_score": best_score}
#                     )
#
#                     if not created:
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#                         if time_diff <= 3600 and best_score > log.match_score:
#                             log.timestamp = current_time
#                             log.match_score = best_score
#                             log.camera = camera
#
#                             if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                                 log.cropped_face.delete(save=False)
#
#                     if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                         try:
#                             _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                             log.cropped_face.save(
#                                 get_attendance_crop_path(student.h_code, camera.name),
#                                 ContentFile(img_encoded.tobytes()),
#                                 save=False
#                             )
#                         except Exception as e:
#                             logger.error(f"Image save error: {e}")
#
#                     log.save()
#                     logger.info(f"✅ Attendance logged for {student.full_name} (Score: {best_score:.4f})")
#
#             if getattr(settings, 'SAVE_ALL_CROPPED_FACES', False):
#                 try:
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                 except Exception as e:
#                     logger.error(f"Error saving debug cropped face: {e}")
#
#     cap.release()
#     logger.info(f"🛌 Stream ended for camera: {camera.name}")




# # The last working version. (Mehdi_2)
# # extras/utils.py
import cv2
import numpy as np
from django.conf import settings
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import logging
import os
import time
from insightface.utils.face_align import norm_crop
from django.core.files.base import ContentFile
import io

from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
from extras.log_utils import get_camera_logger

# === 🔧 PARAMETERS ===
MATCH_THRESHOLD = 0.50  # Lowered for better sensitivity

def is_within_recognition_schedule(current_time, schedule):
    weekday = current_time.strftime('%A')
    return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time

# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map

# Now it falls back to face_embeddings.pkl if individual .npy files are missing.
def load_embeddings(embedding_dir):
    import pickle

    embeddings_map = {}
    npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]

    if npy_files:
        for file in npy_files:
            h_code = file.split('.')[0]
            embedding_path = os.path.join(embedding_dir, file)
            embedding = np.load(embedding_path)
            embeddings_map[h_code] = embedding
    else:
        # Fallback to .pkl file
        pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                embeddings_map = pickle.load(f)
            print("⚠️ Loaded fallback face_embeddings.pkl")
        else:
            print("❌ No embeddings found: .npy files and .pkl are both missing")

    return embeddings_map


def get_attendance_crop_path(student_h_code, camera_name):
    today = datetime.now()
    if not student_h_code or not camera_name:
        raise ValueError("Invalid student_h_code or camera_name in get_attendance_crop_path()")
    return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"

def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
    import uuid
    logger = get_camera_logger(camera.name)

    logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
    logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
    logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")

    cap = cv2.VideoCapture(camera.url)
    if not cap.isOpened():
        logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
        return

    logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")

    use_alignment = hasattr(face_analyzer, 'align_crop')

    if use_alignment:
        logger.info("✅ Face alignment (align_crop) is ENABLED.")
    else:
        logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")

    read_failure_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"⚠️ Failed to read frame from {camera.name}")
            read_failure_count += 1
            if read_failure_count >= 30:
                logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
                break
            time.sleep(10)
            continue
        read_failure_count = 0  # reset on success

        current_time = datetime.now()
        logger.info(f"📸 Processing frame at {current_time} for {camera.name}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb)
        aligned_faces = []

        for face in faces:
            if face.embedding is None or len(face.embedding) == 0:
            # if not face.embedding:
                continue

            debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
            os.makedirs(debug_dir, exist_ok=True)

            base_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            face_box = face.bbox.astype(int)
            x1, y1, x2, y2 = face_box
            cropped_face = rgb[y1:y2, x1:x2]

            save_crop = getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)
            cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")

            if save_crop:
                try:
                    cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                    logger.info(f"✅ Cropped face saved: {cropped_path}")
                except Exception as e:
                    logger.warning(f"❌ Failed saving cropped face: {e}")

            if use_alignment:
                try:
                    aligned_face = face_analyzer.align_crop(rgb, face.kps)
                    aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
                    face.embedding = aligned_embedding

                    aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
                    cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
                    logger.info(f"✅ Aligned face saved: {aligned_path}")

                    combined = cv2.hconcat([
                        cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
                        cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
                    ])
                    compare_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
                    cv2.imwrite(compare_path, combined)
                    logger.info(f"📸 Side-by-side comparison saved: {compare_path}")

                except Exception as e:
                    logger.warning(f"❌ Alignment error: {e}")
                    continue

            aligned_faces.append((face, cropped_face))

        for face, cropped_face in aligned_faces:
            best_score = -1
            best_h_code = None

            for h_code, known_embedding in embeddings_map.items():
                try:
                    score = cosine_similarity([face.embedding], [known_embedding])[0][0]
                except Exception as e:
                    logger.warning(f"❌ Error calculating cosine similarity: {e}")
                    continue

                if score > best_score:
                    best_score = score
                    best_h_code = h_code

            if best_score >= MATCH_THRESHOLD and best_h_code:
                try:
                    best_student = Student.objects.get(h_code=best_h_code)
                except Student.DoesNotExist:
                    logger.warning(f"❌ No student found for H-code: {best_h_code}")
                    continue

                periods = Period.objects.filter(
                    is_active=True,
                    start_time__lte=current_time.time(),
                    end_time__gte=current_time.time()
                )

                if not periods.exists():
                    logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
                    continue

                for period in periods:
                    attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
                    try:
                        log = AttendanceLog.objects.get(
                            student=best_student,
                            period=period,
                            date=current_time.date()
                        )
                        time_diff = abs((current_time - log.timestamp).total_seconds())

                        if time_diff <= attendance_window_seconds and best_score > log.match_score:
                            log.match_score = best_score
                            log.timestamp = current_time
                            log.camera = camera

                            if getattr(settings, "SAVE_CROPPED_IMAGE", True):
                                if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", True) and log.cropped_face:
                                    log.cropped_face.delete(save=False)

                                _, img_encoded = cv2.imencode('.jpg', cropped_face)
                                filename = f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
                                log.cropped_face.save(
                                    filename,
                                    ContentFile(img_encoded.tobytes()),
                                    save=False
                                )

                            log.save()
                            logger.info(f"🔁 Updated log for {best_student.full_name} (Score: {best_score:.4f})")
                        else:
                            logger.info(f"⏳ Already logged: {best_student.full_name} for {period.name}")

                    except AttendanceLog.DoesNotExist:
                        image_content = None

                        if getattr(settings, "SAVE_CROPPED_IMAGE", True):
                            try:
                                success, img_encoded = cv2.imencode('.jpg', cropped_face)
                                if success:
                                    file_name = get_attendance_crop_path(best_student.h_code, camera.name)
                                    image_content = ContentFile(img_encoded.tobytes())
                                    image_content.name = file_name
                                else:
                                    logger.warning("❌ Failed to encode cropped face image.")
                            except Exception as e:
                                logger.error(f"❌ Exception during face image encoding: {e}")

                            if image_content is None:
                                logger.warning(
                                    f"⚠️ Skipping AttendanceLog for {best_student.h_code} due to image encoding failure.")

                        log_data = {
                            'student': best_student,
                            'period': period,
                            'camera': camera,
                            'match_score': best_score,
                            'timestamp': current_time,
                            'date': current_time.date(),
                        }

                        if image_content and getattr(image_content, 'name', None):
                            log_data['cropped_face'] = image_content

                        if not image_content or not getattr(image_content, 'name', None):
                            logger.warning(f"⚠️ No valid cropped image to attach for {best_student.h_code}")

                        AttendanceLog.objects.create(**log_data)
                        logger.info(f"✅ Match: {best_student.full_name} logged (Score: {best_score:.4f})")
            else:
                logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")

    cap.release()
    logger.info(f"🛑 Stopped streaming from camera: {camera.name}")













#
# # extras/utils.py  (Mehdi_1 _ Good but no auto-restart for stream)
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # Lowered for better sensitivity
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# # Now it falls back to face_embeddings.pkl if individual .npy files are missing.
# def load_embeddings(embedding_dir):
#     import pickle
#
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     else:
#         # Fallback to .pkl file
#         pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(pkl_path):
#             with open(pkl_path, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found: .npy files and .pkl are both missing")
#
#     return embeddings_map
#
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     if not student_h_code or not camera_name:
#         raise ValueError("Invalid student_h_code or camera_name in get_attendance_crop_path()")
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(15)
#             continue
#         read_failure_count = 0  # reset on success
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             save_crop = getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#
#             if save_crop:
#                 try:
#                     cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Cropped face saved: {cropped_path}")
#                 except Exception as e:
#                     logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path}")
#
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     compare_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(compare_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {compare_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             old_file_path = log.cropped_face.path if log.cropped_face else None
#
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#
#                             if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                                 _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 filename = f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
#                                 log.cropped_face.save(
#                                     filename,
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#
#                                 # Handle old image based on DELETE_OLD_CROPPED_IMAGE setting
#                                 if old_file_path and os.path.exists(old_file_path):
#                                     if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", True):
#                                         # Move to debug directory
#                                         debug_dir = os.path.join(settings.MEDIA_ROOT, 'attendance_crops',
#                                                                  current_time.strftime('%Y/%m/%d'), 'debug')
#                                         os.makedirs(debug_dir, exist_ok=True)
#                                         new_debug_path = os.path.join(debug_dir, os.path.basename(old_file_path))
#                                         try:
#                                             os.rename(old_file_path, new_debug_path)
#                                             logger.info(f"🗂️ Moved old cropped image to debug: {new_debug_path}")
#                                         except Exception as e:
#                                             logger.warning(f"❌ Failed to move old cropped image to debug: {e}")
#                                     else:
#                                         # Delete the old image
#                                         try:
#                                             os.remove(old_file_path)
#                                             logger.info(f"🗑️ Deleted old cropped image: {old_file_path}")
#                                         except Exception as e:
#                                             logger.warning(f"❌ Failed to delete old cropped image: {e}")
#
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} (Score: {best_score:.4f})")
#                         else:
#                             logger.info(f"⏳ Already logged: {best_student.full_name} for {period.name}")
#
#
#                     except AttendanceLog.DoesNotExist:
#                         image_content = None
#
#                         if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 success, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 if success:
#                                     file_name = get_attendance_crop_path(best_student.h_code, camera.name)
#                                     image_content = ContentFile(img_encoded.tobytes())
#                                     image_content.name = file_name
#                                 else:
#                                     logger.warning("❌ Failed to encode cropped face image.")
#                             except Exception as e:
#                                 logger.error(f"❌ Exception during face image encoding: {e}")
#
#                             if image_content is None:
#                                 logger.warning(
#                                     f"⚠️ Skipping AttendanceLog for {best_student.h_code} due to image encoding failure.")
#
#                         log_data = {
#                             'student': best_student,
#                             'period': period,
#                             'camera': camera,
#                             'match_score': best_score,
#                             'timestamp': current_time,
#                             'date': current_time.date(),
#                         }
#
#                         if image_content and getattr(image_content, 'name', None):
#                             log_data['cropped_face'] = image_content
#
#                         if not image_content or not getattr(image_content, 'name', None):
#                             logger.warning(f"⚠️ No valid cropped image to attach for {best_student.h_code}")
#
#                         AttendanceLog.objects.create(**log_data)
#                         logger.info(f"✅ Match: {best_student.full_name} logged (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")























#
#
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # Lowered for better sensitivity
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     import pickle
#
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     else:
#         pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(pkl_path):
#             with open(pkl_path, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found: .npy files and .pkl are both missing")
#
#     return embeddings_map
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     if not student_h_code or not camera_name:
#         raise ValueError("Invalid student_h_code or camera_name in get_attendance_crop_path()")
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts=0):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"🎥 [START] Processing stream for: {camera.name}")
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     retry_count = 0
#     max_retries = 3
#     cap = None
#     while retry_count < max_retries:
#         cap = cv2.VideoCapture(camera.url)
#
#         if cap.isOpened():
#             break
#         logger.warning(f"⚠️ Attempt {retry_count + 1}: Failed to open stream for {camera.name}. Retrying...")
#         cap.release()
#         retry_count += 1
#         time.sleep(5)
#
#     if not cap or not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     corrupted_frame_count = 0
#     max_corrupted = 10
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None or frame.size == 0:
#             logger.warning(f"⚠️ Invalid frame from {camera.name} (ret={ret}, frame empty={frame is None or frame.size == 0})")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(15)
#             continue
#         read_failure_count = 0
#
#         current_time = datetime.now()
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#                 continue
#
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#             if cropped_face is None or cropped_face.size == 0:
#                 continue
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#                 except:
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_match = None
#             best_score = -1
#             for h_code, ref_emb in embeddings_map.items():
#                 score = cosine_similarity([face.embedding], [ref_emb])[0][0]
#                 if score > best_score:
#                     best_score = score
#                     best_match = h_code
#
#             if best_score >= MATCH_THRESHOLD:
#                 try:
#                     student = Student.objects.get(h_code=best_match, is_active=True)
#                 except Student.DoesNotExist:
#                     continue
#
#                 for sched in schedules:
#                     if is_within_recognition_schedule(current_time, sched):
#                         period = sched.period
#                         log, created = AttendanceLog.objects.get_or_create(
#                             student=student,
#                             period=period,
#                             date=current_time.date(),
#                             defaults={'camera': camera}
#                         )
#
#                         if not created:
#                             log.camera = camera  # Ensure camera is updated
#
#                         if getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False) and log.cropped_face:
#                             try:
#                                 if os.path.exists(log.cropped_face.path):
#                                     os.remove(log.cropped_face.path)
#                             except Exception as e:
#                                 logger.warning(f"Could not delete old cropped image: {e}")
#
#                         if getattr(settings, 'SAVE_CROPPED_IMAGE', True):
#                             _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                             log.cropped_face.save(
#                                 get_attendance_crop_path(student.h_code, camera.name),
#                                 ContentFile(img_encoded.tobytes())
#                             )
#                         log.save()
#
#                         logger.info(f"✅ Attendance logged for {student.h_code} in {period.name} via {camera.name}")
#
#             if getattr(settings, 'SAVE_ALL_CROPPED_FACES', False):
#                 debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#                 os.makedirs(debug_dir, exist_ok=True)
#                 debug_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                 debug_path = os.path.join(debug_dir, debug_filename)
#                 cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#
    # cap.release()
    # logger.info(f"🚩 Stopped streaming from camera: {camera.name}")
    # logger.info(f"🔁 Attempting auto-restart for {camera.name} in 10 seconds...")
    # time.sleep(10)
    # if restart_attempts < 5:
    #     process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts + 1)
    # else:
    #     logger.error(f"🛑 Max auto-restart attempts reached for {camera.name}. Giving up.")
#
#




# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # Lowered for better sensitivity
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     import pickle
#
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     else:
#         pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(pkl_path):
#             with open(pkl_path, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found: .npy files and .pkl are both missing")
#
#     return embeddings_map
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     if not student_h_code or not camera_name:
#         raise ValueError("Invalid student_h_code or camera_name in get_attendance_crop_path()")
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts=0):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     retry_count = 0
#     max_retries = 3
#     cap = None
#     while retry_count < max_retries:
#         cap = cv2.VideoCapture(camera.url)
#
#         if cap.isOpened():
#             break
#         logger.warning(f"⚠️ Attempt {retry_count + 1}: Failed to open stream for {camera.name}. Retrying...")
#         cap.release()
#         retry_count += 1
#         time.sleep(5)
#
#     if not cap or not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     corrupted_frame_count = 0
#     max_corrupted = 10
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None or frame.size == 0:
#             logger.warning(f"⚠️ Invalid frame from {camera.name} (ret={ret}, frame empty={frame is None or frame.size == 0})")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(15)
#             continue
#         read_failure_count = 0
#
#         current_time = datetime.now()
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#                 continue
#
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#             if cropped_face is None or cropped_face.size == 0:
#                 continue
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#                 except:
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_match = None
#             best_score = -1
#             for h_code, ref_emb in embeddings_map.items():
#                 score = cosine_similarity([face.embedding], [ref_emb])[0][0]
#                 if score > best_score:
#                     best_score = score
#                     best_match = h_code
#
#             if best_score >= MATCH_THRESHOLD:
#                 try:
#                     student = Student.objects.get(h_code=best_match, is_active=True)
#                 except Student.DoesNotExist:
#                     continue
#
#                 for sched in schedules:
#                     if is_within_recognition_schedule(current_time, sched):
#                         period = sched.period
#                         log, created = AttendanceLog.objects.get_or_create(
#                             student=student,
#                             period=period,
#                             date=current_time.date(),
#                             defaults={'camera': camera}
#                         )
#
#                         if not created and getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False):
#                             if log.cropped_face and os.path.exists(log.cropped_face.path):
#                                 os.remove(log.cropped_face.path)
#
#                         if not log.cropped_face and getattr(settings, 'SAVE_CROPPED_IMAGE', True):
#                             _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                             log.cropped_face.save(
#                                 get_attendance_crop_path(student.h_code, camera.name),
#                                 ContentFile(img_encoded.tobytes())
#                             )
#                             log.save()
#
#             if getattr(settings, 'SAVE_ALL_CROPPED_FACES', False):
#                 debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#                 os.makedirs(debug_dir, exist_ok=True)
#                 debug_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                 debug_path = os.path.join(debug_dir, debug_filename)
#                 cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#
#     cap.release()
#     logger.info(f"🚩 Stopped streaming from camera: {camera.name}")
#     logger.info(f"🔁 Attempting auto-restart for {camera.name} in 10 seconds...")
#     time.sleep(10)
#     if restart_attempts < 5:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts + 1)
#     else:
#         logger.error(f"🛑 Max auto-restart attempts reached for {camera.name}. Giving up.")



# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # Lowered for better sensitivity
#
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
#
# def load_embeddings(embedding_dir):
#     import pickle
#
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     else:
#         pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(pkl_path):
#             with open(pkl_path, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found: .npy files and .pkl are both missing")
#
#     return embeddings_map
#
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     if not student_h_code or not camera_name:
#         raise ValueError("Invalid student_h_code or camera_name in get_attendance_crop_path()")
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts=0):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     retry_count = 0
#     max_retries = 3
#     cap = None
#     while retry_count < max_retries:
#         cap = cv2.VideoCapture(camera.url)
#
#         if camera.resolution:
#             try:
#                 width, height = map(int, camera.resolution.lower().split('x'))
#                 cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
#                 cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
#                 logger.info(f"📏 Camera resolution set to {width}x{height} for {camera.name}")
#             except Exception as e:
#                 logger.warning(f"⚠️ Failed to set resolution: {e}")
#
#         if cap.isOpened():
#             break
#         logger.warning(f"⚠️ Attempt {retry_count + 1}: Failed to open stream for {camera.name}. Retrying...")
#         cap.release()
#         retry_count += 1
#         time.sleep(5)
#
#     if not cap or not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     corrupted_frame_count = 0
#     max_corrupted = 10
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None or frame.size == 0:
#             logger.warning(f"⚠️ Invalid frame from {camera.name} (ret={ret}, frame empty={frame is None or frame.size == 0})")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(15)
#             continue
#         read_failure_count = 0
#
#         current_time = datetime.now()
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#                 continue
#
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#             if cropped_face is None or cropped_face.size == 0:
#                 continue
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#                 except:
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_match = None
#             best_score = -1
#             for h_code, ref_emb in embeddings_map.items():
#                 score = cosine_similarity([face.embedding], [ref_emb])[0][0]
#                 if score > best_score:
#                     best_score = score
#                     best_match = h_code
#
#             if best_score >= MATCH_THRESHOLD:
#                 try:
#                     student = Student.objects.get(h_code=best_match, is_active=True)
#                 except Student.DoesNotExist:
#                     continue
#
#                 for sched in schedules:
#                     if is_within_recognition_schedule(current_time, sched):
#                         period = sched.period
#                         log, created = AttendanceLog.objects.get_or_create(
#                             student=student,
#                             period=period,
#                             date=current_time.date(),
#                             defaults={'camera': camera}
#                         )
#
#                         if not created and getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False):
#                             if log.cropped_face and os.path.exists(log.cropped_face.path):
#                                 os.remove(log.cropped_face.path)
#
#                         if not log.cropped_face:
#                             _, img_encoded = cv2.imencode('.jpg', cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                             log.cropped_face.save(
#                                 get_attendance_crop_path(student.h_code, camera.name),
#                                 ContentFile(img_encoded.tobytes())
#                             )
#                             log.save()
#
#     cap.release()
#     logger.info(f"🚩 Stopped streaming from camera: {camera.name}")
#     logger.info(f"🔁 Attempting auto-restart for {camera.name} in 10 seconds...")
#     time.sleep(10)
#     if restart_attempts < 5:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts + 1)
#     else:
#         logger.error(f"🛑 Max auto-restart attempts reached for {camera.name}. Giving up.")
#
#
#



# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # Lowered for better sensitivity
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     import pickle
#
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     else:
#         pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(pkl_path):
#             with open(pkl_path, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found: .npy files and .pkl are both missing")
#
#     return embeddings_map
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     if not student_h_code or not camera_name:
#         raise ValueError("Invalid student_h_code or camera_name in get_attendance_crop_path()")
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts=0):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     retry_count = 0
#     max_retries = 3
#     cap = None
#     while retry_count < max_retries:
#         cap = cv2.VideoCapture(camera.url)
#         if cap.isOpened():
#             break
#         logger.warning(f"⚠️ Attempt {retry_count + 1}: Failed to open stream for {camera.name}. Retrying...")
#         cap.release()
#         retry_count += 1
#         time.sleep(5)
#
#     if not cap or not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     corrupted_frame_count = 0
#     max_corrupted = 10
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None or frame.size == 0:
#             logger.warning(f"⚠️ Invalid frame from {camera.name} (ret={ret}, frame empty={frame is None or frame.size == 0})")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(15)
#             continue
#         read_failure_count = 0
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             if cropped_face is None or cropped_face.size == 0 or cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
#                 corrupted_frame_count += 1
#                 logger.warning(f"⚠️ Detected corrupted cropped face (#{corrupted_frame_count}) from {camera.name}")
#                 if corrupted_frame_count >= max_corrupted:
#                     logger.error(f"🔥 Too many corrupted frames from {camera.name}. Restarting stream...")
#                     break
#                 continue
#             else:
#                 corrupted_frame_count = 0
#
#             save_crop = getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#
#             if save_crop:
#                 try:
#                     cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Cropped face saved: {cropped_path}")
#                 except Exception as e:
#                     logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path}")
#
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     compare_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(compare_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {compare_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             if cropped_face is None or cropped_face.size == 0:
#                 logger.warning("⚠️ Skipping empty cropped face before matching or saving log.")
#                 continue
#
#             # rest of your matching and logging code here
#
#     cap.release()
#     logger.info(f"🚩 Stopped streaming from camera: {camera.name}")
#     logger.info(f"🔁 Attempting auto-restart for {camera.name} in 10 seconds...")
#     time.sleep(10)
#     if restart_attempts < 5:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts + 1)
#     else:
#         logger.error(f"🛑 Max auto-restart attempts reached for {camera.name}. Giving up.")
#
#
#







# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # Lowered for better sensitivity
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     import pickle
#
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     else:
#         pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(pkl_path):
#             with open(pkl_path, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found: .npy files and .pkl are both missing")
#
#     return embeddings_map
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     if not student_h_code or not camera_name:
#         raise ValueError("Invalid student_h_code or camera_name in get_attendance_crop_path()")
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts=0):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     retry_count = 0
#     max_retries = 3
#     cap = None
#     while retry_count < max_retries:
#         cap = cv2.VideoCapture(camera.url)
#         if cap.isOpened():
#             break
#         logger.warning(f"⚠️ Attempt {retry_count + 1}: Failed to open stream for {camera.name}. Retrying...")
#         cap.release()
#         retry_count += 1
#         time.sleep(5)
#
#     if not cap or not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     corrupted_frame_count = 0
#     max_corrupted = 10
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None or frame.size == 0:
#             logger.warning(f"⚠️ Invalid frame from {camera.name} (ret={ret}, frame empty={frame is None or frame.size == 0})")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(15)
#             continue
#         read_failure_count = 0
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             if cropped_face is None or cropped_face.size == 0 or cropped_face.shape[0] == 0 or cropped_face.shape[1] == 0:
#                 corrupted_frame_count += 1
#                 logger.warning(f"⚠️ Detected corrupted cropped face (#{corrupted_frame_count}) from {camera.name}")
#                 if corrupted_frame_count >= max_corrupted:
#                     logger.error(f"🔥 Too many corrupted frames from {camera.name}. Restarting stream...")
#                     break
#                 continue
#             else:
#                 corrupted_frame_count = 0
#                 logger.warning(f"⚠️ Cropped face is empty. Skipping this face.")
#                 continue
#
#             save_crop = getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#
#             if save_crop:
#                 try:
#                     cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Cropped face saved: {cropped_path}")
#                 except Exception as e:
#                     logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path}")
#
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     compare_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(compare_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {compare_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             if cropped_face is None or cropped_face.size == 0:
#                 logger.warning("⚠️ Skipping empty cropped face before matching or saving log.")
#                 continue
#
#             # rest of your matching and logging code here
#
#     cap.release()
#     logger.info(f"🚩 Stopped streaming from camera: {camera.name}")
#     logger.info(f"🔁 Attempting auto-restart for {camera.name} in 10 seconds...")
#     time.sleep(10)
#     if restart_attempts < 5:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts + 1)
#     else:
#         logger.error(f"🛑 Max auto-restart attempts reached for {camera.name}. Giving up.")
#
#





# # extras/utils.py  (Good with 10 Sec auto-restart for stream, to be set at the end of the code.)

# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # Lowered for better sensitivity
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     import pickle
#
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     else:
#         pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(pkl_path):
#             with open(pkl_path, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found: .npy files and .pkl are both missing")
#
#     return embeddings_map
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     if not student_h_code or not camera_name:
#         raise ValueError("Invalid student_h_code or camera_name in get_attendance_crop_path()")
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts=0):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     retry_count = 0
#     max_retries = 3
#     cap = None
#     while retry_count < max_retries:
#         cap = cv2.VideoCapture(camera.url)
#         if cap.isOpened():
#             break
#         logger.warning(f"⚠️ Attempt {retry_count + 1}: Failed to open stream for {camera.name}. Retrying...")
#         cap.release()
#         retry_count += 1
#         time.sleep(5)
#
#     if not cap or not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None or frame.size == 0:
#             logger.warning(f"⚠️ Invalid frame from {camera.name} (ret={ret}, frame empty={frame is None or frame.size == 0})")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(15)
#             continue
#         read_failure_count = 0
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             if cropped_face is None or cropped_face.size == 0:
#                 logger.warning(f"⚠️ Cropped face is empty. Skipping this face.")
#                 continue
#
#             save_crop = getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#
#             if save_crop:
#                 try:
#                     cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Cropped face saved: {cropped_path}")
#                 except Exception as e:
#                     logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path}")
#
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     compare_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(compare_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {compare_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             if cropped_face is None or cropped_face.size == 0:
#                 logger.warning("⚠️ Skipping empty cropped face before matching or saving log.")
#                 continue
#
#             # rest of your matching and logging code here
#
#     cap.release()
#     logger.info(f"🚩 Stopped streaming from camera: {camera.name}")
#     logger.info(f"🔁 Attempting auto-restart for {camera.name} in 10 seconds...")
#     time.sleep(10)
#     if restart_attempts < 5:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map, restart_attempts + 1)
#     else:
#         logger.error(f"🛑 Max auto-restart attempts reached for {camera.name}. Giving up.")










# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(10)
#             continue
#         read_failure_count = 0  # reset on success
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#             # if not face.embedding:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             save_crop = getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#
#             if save_crop:
#                 try:
#                     cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Cropped face saved: {cropped_path}")
#                 except Exception as e:
#                     logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path}")
#
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     compare_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(compare_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {compare_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         # if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                         #     log.match_score = best_score
#                         #     log.timestamp = current_time
#                         #     log.camera = camera
#                         #
#                         #     if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                         #         if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", True) and log.cropped_face:
#                         #             log.cropped_face.delete(save=False)
#                         #
#                         #         _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                         #         filename = f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
#                         #         log.cropped_face.save(
#                         #             filename,
#                         #             ContentFile(img_encoded.tobytes()),
#                         #             save=False
#                         #         )
#                         #
#                         #     log.save()
#                         #     logger.info(f"🔁 Updated log for {best_student.full_name} (Score: {best_score:.4f})")
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#
#                             if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                                 if log.cropped_face:
#                                     old_file_path = log.cropped_face.path
#                                     log.cropped_face.delete(save=False)
#                                     if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False):
#                                         try:
#                                             if os.path.exists(old_file_path):
#                                                 os.remove(old_file_path)
#                                                 logger.info(f"🗑️ Deleted old cropped face file: {old_file_path}")
#                                         except Exception as e:
#                                             logger.warning(f"❌ Failed to delete old cropped image file: {e}")
#
#                                 _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 filename = f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
#                                 log.cropped_face.save(
#                                     filename,
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} (Score: {best_score:.4f})")
#
#                         else:
#                             logger.info(f"⏳ Already logged: {best_student.full_name} for {period.name}")
#
#                     except AttendanceLog.DoesNotExist:
#                         image_content = None
#
#                         if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 success, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 if success:
#                                     file_name = get_attendance_crop_path(best_student.h_code, camera.name)
#                                     image_content = ContentFile(img_encoded.tobytes())
#                                     image_content.name = file_name
#                                 else:
#                                     logger.warning("❌ Failed to encode cropped face image.")
#                             except Exception as e:
#                                 logger.error(f"❌ Exception during face image encoding: {e}")
#
#                             if image_content is None:
#                                 logger.warning(
#                                     f"⚠️ Skipping AttendanceLog for {best_student.h_code} due to image encoding failure.")
#
#                         log_data = {
#                             'student': best_student,
#                             'period': period,
#                             'camera': camera,
#                             'match_score': best_score,
#                             'timestamp': current_time,
#                             'date': current_time.date(),
#                         }
#
#                         if image_content and getattr(image_content, 'name', None):
#                             log_data['cropped_face'] = image_content
#
#                         if not image_content or not getattr(image_content, 'name', None):
#                             logger.warning(f"⚠️ No valid cropped image to attach for {best_student.h_code}")
#
#                         AttendanceLog.objects.create(**log_data)
#                         logger.info(f"✅ Match: {best_student.full_name} logged (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")



#
#
# # extras/utils.py
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.55  # Lowered for better sensitivity
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# # def load_embeddings(embedding_dir):
# #     embeddings_map = {}
# #     for file in os.listdir(embedding_dir):
# #         if file.endswith('.npy'):
# #             h_code = file.split('.')[0]
# #             embedding_path = os.path.join(embedding_dir, file)
# #             embedding = np.load(embedding_path)
# #             embeddings_map[h_code] = embedding
# #     return embeddings_map
#
# # Now it falls back to face_embeddings.pkl if individual .npy files are missing.
# def load_embeddings(embedding_dir):
#     import pickle
#
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     else:
#         # Fallback to .pkl file
#         pkl_path = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(pkl_path):
#             with open(pkl_path, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found: .npy files and .pkl are both missing")
#
#     return embeddings_map
#
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     if not student_h_code or not camera_name:
#         raise ValueError("Invalid student_h_code or camera_name in get_attendance_crop_path()")
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(10)
#             continue
#         read_failure_count = 0  # reset on success
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#             # if not face.embedding:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             save_crop = getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#
#             if save_crop:
#                 try:
#                     cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Cropped face saved: {cropped_path}")
#                 except Exception as e:
#                     logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path}")
#
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     compare_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(compare_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {compare_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#
#                             if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                                 if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", True) and log.cropped_face:
#                                     log.cropped_face.delete(save=False)
#
#                                 _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 filename = f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
#                                 log.cropped_face.save(
#                                     filename,
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} (Score: {best_score:.4f})")
#                         else:
#                             logger.info(f"⏳ Already logged: {best_student.full_name} for {period.name}")
#
#                     except AttendanceLog.DoesNotExist:
#                         image_content = None
#
#                         if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 success, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 if success:
#                                     file_name = get_attendance_crop_path(best_student.h_code, camera.name)
#                                     image_content = ContentFile(img_encoded.tobytes())
#                                     image_content.name = file_name
#                                 else:
#                                     logger.warning("❌ Failed to encode cropped face image.")
#                             except Exception as e:
#                                 logger.error(f"❌ Exception during face image encoding: {e}")
#
#                             if image_content is None:
#                                 logger.warning(
#                                     f"⚠️ Skipping AttendanceLog for {best_student.h_code} due to image encoding failure.")
#
#                         log_data = {
#                             'student': best_student,
#                             'period': period,
#                             'camera': camera,
#                             'match_score': best_score,
#                             'timestamp': current_time,
#                             'date': current_time.date(),
#                         }
#
#                         if image_content and getattr(image_content, 'name', None):
#                             log_data['cropped_face'] = image_content
#
#                         if not image_content or not getattr(image_content, 'name', None):
#                             logger.warning(f"⚠️ No valid cropped image to attach for {best_student.h_code}")
#
#                         AttendanceLog.objects.create(**log_data)
#                         logger.info(f"✅ Match: {best_student.full_name} logged (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")


#
#
# # extras/utils.py
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.50  # Lowered for better sensitivity
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     if not student_h_code or not camera_name:
#         raise ValueError("Invalid student_h_code or camera_name in get_attendance_crop_path()")
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', '???')}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', '???')}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', '???')}")
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     # while True:
#     #     ret, frame = cap.read()
#     #     if not ret:
#     #         logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#     #         time.sleep(10)
#     #         break
#
#     read_failure_count = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             read_failure_count += 1
#             if read_failure_count >= 30:
#                 logger.error(f"🔥 Giving up on camera {camera.name} after 30 failed attempts.")
#                 break
#             time.sleep(10)
#             continue
#         read_failure_count = 0  # reset on success
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None or len(face.embedding) == 0:
#             # if not face.embedding:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             save_crop = getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#
#             if save_crop:
#                 try:
#                     cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Cropped face saved: {cropped_path}")
#                 except Exception as e:
#                     logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path}")
#
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     compare_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(compare_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {compare_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#
#                             if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                                 if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", True) and log.cropped_face:
#                                     log.cropped_face.delete(save=False)
#
#                                 _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 # log.cropped_face.save(
#                                 #     get_attendance_crop_path(best_student.h_code, camera.name),
#                                 #     ContentFile(img_encoded.tobytes()),
#                                 #     save=False
#                                 # )
#
#                                 filename = f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
#                                 log.cropped_face.save(
#                                     filename,
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} (Score: {best_score:.4f})")
#                         else:
#                             logger.info(f"⏳ Already logged: {best_student.full_name} for {period.name}")
#
#                     except AttendanceLog.DoesNotExist:
#                         image_content = None
#
#                         if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 success, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 if success:
#                                     file_name = get_attendance_crop_path(best_student.h_code, camera.name)
#                                     image_content = ContentFile(img_encoded.tobytes())
#                                     image_content.name = file_name
#                                 else:
#                                     logger.warning("❌ Failed to encode cropped face image.")
#                             except Exception as e:
#                                 logger.error(f"❌ Exception during face image encoding: {e}")
#
#                             if image_content is None:
#                                 logger.warning(
#                                     f"⚠️ Skipping AttendanceLog for {best_student.h_code} due to image encoding failure.")
#
#                         log_data = {
#                             'student': best_student,
#                             'period': period,
#                             'camera': camera,
#                             'match_score': best_score,
#                             'timestamp': current_time,
#                             'date': current_time.date(),
#                         }
#
#                         if image_content and getattr(image_content, 'name', None):
#                             log_data['cropped_face'] = image_content
#
#                         if not image_content or not getattr(image_content, 'name', None):
#                             logger.warning(f"⚠️ No valid cropped image to attach for {best_student.h_code}")
#
#                         AttendanceLog.objects.create(**log_data)
#                         logger.info(f"✅ Match: {best_student.full_name} logged (Score: {best_score:.4f})")
#
#                         # AttendanceLog.objects.create(
#                         #     student=best_student,
#                         #     period=period,
#                         #     camera=camera,
#                         #     match_score=best_score,
#                         #     timestamp=current_time,
#                         #     date=current_time.date(),
#                         #     cropped_face=image_content
#                         # )
#                         # logger.info(f"✅ Match: {best_student.full_name} logged (Score: {best_score:.4f})")
#
#                     # except AttendanceLog.DoesNotExist:
#                     #     # image_content = None
#                     #     # image_content = ContentFile(
#                     #     #     img_encoded.tobytes(),
#                     #     #     name=get_attendance_crop_path(...)
#                     #     # )
#                     #
#                     #     if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                     #         # _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                     #         # image_content = ContentFile(img_encoded.tobytes(), name=get_attendance_crop_path(best_student.h_code, camera.name))
#                     #
#                     #         try:
#                     #             success, img_encoded = cv2.imencode('.jpg', cropped_face)
#                     #             # if success:
#                     #             #     image_content = ContentFile(
#                     #             #         img_encoded.tobytes(),
#                     #             #         name=get_attendance_crop_path(best_student.h_code, camera.name)
#                     #             #     )
#                     #             if success:
#                     #                 file_name = get_attendance_crop_path(best_student.h_code, camera.name)
#                     #                 image_content = ContentFile(img_encoded.tobytes())
#                     #                 image_content.name = file_name
#                     #             else:
#                     #                 logger.warning("❌ Failed to encode cropped face image.")
#                     #         except Exception as e:
#                     #             logger.error(f"❌ Exception during face image encoding: {e}")
#                     #
#                     #         if image_content is None:
#                     #             logger.warning(
#                     #                 f"⚠️ Skipping AttendanceLog for {best_student.h_code} due to image encoding failure.")
#                     #
#                     #     AttendanceLog.objects.create(
#                     #         student=best_student,
#                     #         period=period,
#                     #         camera=camera,
#                     #         match_score=best_score,
#                     #         timestamp=current_time,
#                     #         date=current_time.date(),
#                     #         cropped_face=image_content
#                     #     )
#                     #     logger.info(f"✅ Match: {best_student.full_name} logged (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")
#



#
# # extras/utils.py
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.50  # Lowered for better sensitivity
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(10)
#             break
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             save_crop = getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#
#             if save_crop:
#                 try:
#                     cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Cropped face saved: {cropped_path}")
#                 except Exception as e:
#                     logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path}")
#
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     compare_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(compare_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {compare_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         for face, cropped_face in aligned_faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#
#                             if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                                 if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", True) and log.cropped_face:
#                                     log.cropped_face.delete(save=False)
#
#                                 _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 log.cropped_face.save(
#                                     get_attendance_crop_path(best_student.h_code, camera.name),
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#
#                                 # _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 # filename = f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg"
#                                 # log.cropped_face.save(
#                                 #     filename,
#                                 #     ContentFile(img_encoded.tobytes()),
#                                 #     save=False
#                                 # )
#
#                                 # log.cropped_face.save(
#                                 #     get_attendance_crop_path(best_student.h_code, camera.name),
#                                 #     ContentFile(img_encoded.tobytes()),
#                                 #     save=False
#                                 # )
#
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} (Score: {best_score:.4f})")
#                         else:
#                             logger.info(f"⏳ Already logged: {best_student.full_name} for {period.name}")
#                     except AttendanceLog.DoesNotExist:
#                         image_content = None
#                         if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                             image_content = ContentFile(img_encoded.tobytes(), name=get_attendance_crop_path(best_student.h_code, camera.name))
#
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date(),
#                             cropped_face=image_content
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")
#



# # extras/utils.py
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from django.core.files.base import ContentFile
# import uuid
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.50
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding = np.load(os.path.join(embedding_dir, file))
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     cap = cv2.VideoCapture(camera.url)
#
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     save_all = getattr(settings, "SAVE_ALL_CROPPED_FACES", True)
#     delete_old = getattr(settings, "DELETE_OLD_CROPPED_IMAGE", True)
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(10)
#             break
#
#         current_time = datetime.now()
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             base_filename = f"{current_time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             x1, y1, x2, y2 = face.bbox.astype(int)
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             # Save all cropped faces to debug directory
#             if save_all:
#                 debug_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#                 try:
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"📥 Saved cropped face to: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"❌ Failed to save cropped face: {e}")
#
#             # Alignment (if available)
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue
#
#             # MATCHING
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"❌ Cosine error: {e}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 for period in periods:
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         if best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#
#                             if delete_old and log.cropped_face:
#                                 log.cropped_face.delete(save=False)
#
#                             _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                             log.cropped_face.save(
#                                 f"{student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg",
#                                 ContentFile(img_encoded.tobytes()),
#                                 save=False
#                             )
#                             log.save()
#                             logger.info(f"🔁 Updated log for {student.full_name} with new face (score: {best_score:.4f})")
#                     except AttendanceLog.DoesNotExist:
#                         _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                         log = AttendanceLog.objects.create(
#                             student=student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date()
#                         )
#                         log.cropped_face.save(
#                             f"{student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg",
#                             ContentFile(img_encoded.tobytes()),
#                             save=True
#                         )
#                         logger.info(f"✅ Created log for {student.full_name} (score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from: {camera.name}")




# # extras/utils.py
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.50  # Lowered for better sensitivity
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(10) # If no stream (for 10 seconds) then Break
#             break
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#
#             # Save original cropped face
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#
#             save_crop = getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)
#             cropped_path = None  # Will only be defined if saving is allowed
#
#             if save_crop:
#                 cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#                 try:
#                     cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Cropped face saved: {cropped_path}")
#                 except Exception as e:
#                     logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             aligned_path = None  # default fallback if not aligned
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path} (size: {aligned_face.shape[0]}x{aligned_face.shape[1]})")
#
#                     # === Create side-by-side comparison
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     combined_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(combined_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {combined_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error, skipping aligned face: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         # === MATCHING LOGIC
#         for face, cropped_face in aligned_faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 will_be_used = True
#                 # If SAVE_ALL_CROPPED_FACES is False, we only save now because it's used
#                 if not save_crop:
#                     cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#                     try:
#                         cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                         logger.info(f"✅ Cropped face saved for recognized student: {cropped_path}")
#                     except Exception as e:
#                         logger.warning(f"❌ Failed saving matched cropped face: {e}")
#                     # try:
#                     #     cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     #     logger.info(f"✅ Cropped face saved for recognized student: {cropped_path}")
#                     # except Exception as e:
#                     #     logger.warning(f"❌ Failed saving matched cropped face: {e}")
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#
#                             if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                                 if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", True) and log.cropped_face:
#                                     log.cropped_face.delete(save=False)
#
#                                 _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                                 log.cropped_face.save(
#                                     f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg",
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} with aligned face (score: {best_score:.4f}) ✅")
#                         else:
#                             logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#                     except AttendanceLog.DoesNotExist:
#                         image_content = None
#                         if getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                             image_content = ContentFile(img_encoded.tobytes(), name=f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg")
#
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date(),
#                             cropped_face=image_content
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged (aligned face) for period {period.name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")


# # extras/utils.py
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
# from django.core.files.base import ContentFile
# import io
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.50  # Lowered for better sensitivity
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(10) # If no stream (for 10 seconds) then Break
#             break
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#
#             # Save original cropped face
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#             try:
#                 cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                 logger.info(f"✅ Cropped face saved: {cropped_path}")
#             except Exception as e:
#                 logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             aligned_path = None  # default fallback if not aligned
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path} (size: {aligned_face.shape[0]}x{aligned_face.shape[1]})")
#
#                     # === Create side-by-side comparison
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     combined_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(combined_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {combined_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error, skipping aligned face: {e}")
#                     continue
#
#             aligned_faces.append((face, cropped_face))
#
#         # === MATCHING LOGIC
#         for face, cropped_face in aligned_faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#
#                             _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                             log.cropped_face.save(
#                                 f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg",
#                                 ContentFile(img_encoded.tobytes()),
#                                 save=False
#                             )
#
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} with aligned face (score: {best_score:.4f}) ✅")
#                         else:
#                             logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#                     except AttendanceLog.DoesNotExist:
#                         _, img_encoded = cv2.imencode('.jpg', cropped_face)
#                         image_content = ContentFile(img_encoded.tobytes(), name=f"{best_student.h_code}_{camera.name}_{current_time.strftime('%Y%m%d_%H%M%S')}.jpg")
#
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date(),
#                             cropped_face=image_content
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged (aligned face) for period {period.name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")



# 20250502 (Working_no cropped_face auto upload to AttendanceLog)
# # extras/utils.py
# import cv2
# import numpy as np
# from django.conf import settings
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime
# import logging
# import os
# import time
# from insightface.utils.face_align import norm_crop
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from extras.log_utils import get_camera_logger
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.50  # Lowered for better sensitivity
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(10) # If no stream (for 10 seconds) then Break
#             break
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#
#             # Save original cropped face
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#             try:
#                 cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                 logger.info(f"✅ Cropped face saved: {cropped_path}")
#             except Exception as e:
#                 logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path} (size: {aligned_face.shape[0]}x{aligned_face.shape[1]})")
#
#                     # === Create side-by-side comparison
#                     combined = cv2.hconcat([
#                         cv2.resize(cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR), (112, 112)),
#                         cv2.resize(cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR), (112, 112))
#                     ])
#                     combined_path = os.path.join(debug_dir, f"{base_filename}_compare.jpg")
#                     cv2.imwrite(combined_path, combined)
#                     logger.info(f"📸 Side-by-side comparison saved: {combined_path}")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error, skipping aligned face: {e}")
#                     continue
#
#             aligned_faces.append(face)
#
#         faces = aligned_faces
#
#         # === MATCHING LOGIC
#         for face in faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} with aligned face (score: {best_score:.4f}) ✅")
#                         else:
#                             logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#                     except AttendanceLog.DoesNotExist:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date()
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged (aligned face) for period {period.name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")
#
#





# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             base_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             try:
#                 cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                 logger.info(f"✅ Cropped face saved: {cropped_path}")
#             except Exception as e:
#                 logger.warning(f"❌ Failed saving cropped face: {e}")
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"✅ Aligned face saved: {aligned_path} (size: {aligned_face.shape[0]}x{aligned_face.shape[1]})")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error, skipping saving aligned face: {e}")
#                     continue
#
#             aligned_faces.append(face)
#
#         faces = aligned_faces
#
#         # === MATCHING LOGIC (same as yours)
#         for face in faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} with better score: {best_score:.4f}")
#                         else:
#                             logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#                     except AttendanceLog.DoesNotExist:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date()
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged for period {period.name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")
#







# # This is working, it uses norm_crop, it saves the cropped face and the aligned face
# #
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid
#     logger = get_camera_logger(camera.name)
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#
#             # Build a base filename once
#             base_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
#
#             # Save original bbox crop
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             cropped_path = os.path.join(debug_dir, f"{base_filename}.jpg")
#             cv2.imwrite(cropped_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#
#                     # Save aligned face with _aligned suffix
#                     aligned_path = os.path.join(debug_dir, f"{base_filename}_aligned.jpg")
#                     cv2.imwrite(aligned_path, cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR))
#                     logger.info("✅ aligned face with _aligned suffix saved.")
#
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue  # Skip badly aligned faces
#
#             aligned_faces.append(face)
#
#         faces = aligned_faces
#
#         # === MATCHING LOGIC
#         for face in faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} with better score: {best_score:.4f}")
#                         else:
#                             logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#                     except AttendanceLog.DoesNotExist:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date()
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged for period {period.name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")
#
#



# This is working, it uses norm_crop, it saves the cropped face only
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid  # Ensure this import
#     logger = get_camera_logger(camera.name)
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     # === Check if align_crop is available
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         aligned_faces = []
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             if use_alignment:
#                 try:
#                     aligned_face = face_analyzer.align_crop(rgb, face.kps)
#                     aligned_embedding = face_analyzer.get(aligned_face)[0].embedding
#                     face.embedding = aligned_embedding
#                 except Exception as e:
#                     logger.warning(f"❌ Alignment error: {e}")
#                     continue  # Skip badly aligned faces
#
#             aligned_faces.append(face)
#
#             # Save cropped face for debugging
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#             debug_filename = os.path.join(debug_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg")
#             cv2.imwrite(debug_filename, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#
#         # Replace faces with aligned_faces
#         faces = aligned_faces
#
#         # === MATCHING LOGIC
#         for face in faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} with better score: {best_score:.4f}")
#                         else:
#                             logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#                     except AttendanceLog.DoesNotExist:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date()
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged for period {period.name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")



# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     import uuid  # Import here or at top of file
#     logger = get_camera_logger(camera.name)  # <-- 🔥 Per camera logger here
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#         # if not ret:
#         #     logger.warning(f"⚠️ Failed to read frame from {camera.name}, retrying...")
#         #     time.sleep(2)
#         #     continue  # <--- instead of break
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         # Align faces and save cropped faces
#         aligned_faces = []
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             # Align the face
#             aligned_face = face_analyzer.align_crop(rgb, face.kps)
#             face.embedding = face_analyzer.get(aligned_face)[0].embedding
#             aligned_faces.append(face)
#
#             # Save cropped face for debugging
#             face_box = face.bbox.astype(int)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#             debug_filename = os.path.join(debug_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg")
#             cv2.imwrite(debug_filename, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#
#         faces = aligned_faces  # <<< ✅ THIS MUST BE AFTER for-loop is finished
#
#         # --- Face matching ---
#         for face in faces:
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} with better score: {best_score:.4f}")
#                         else:
#                             logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#                     except AttendanceLog.DoesNotExist:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date()
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged for period {period.name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")


#  $ This works but no align_crop function in insightface
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)  # <-- 🔥 Per camera logger here
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = datetime.now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         # faces = face_analyzer.get(rgb)
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         aligned_faces = []
#         for face in faces:
#             if face.embedding is None:
#                 continue
#             # Align the face using its landmarks
#             aligned_face = face_analyzer.align_crop(rgb, face.kps)
#             face.embedding = face_analyzer.get(aligned_face)[0].embedding
#             aligned_faces.append(face)
#
#             face_box = face.bbox.astype(int)  # (x1, y1, x2, y2)
#             x1, y1, x2, y2 = face_box
#             cropped_face = rgb[y1:y2, x1:x2]
#
#             # Save the cropped face temporarily for debugging
#             debug_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'debug_faces', camera.name)
#             os.makedirs(debug_dir, exist_ok=True)
#             debug_filename = os.path.join(debug_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}.jpg")
#             cv2.imwrite(debug_filename, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#
#             faces = aligned_faces
#
#
#
#         # for face in faces:
#         #     if face.embedding is None:
#         #         continue
#
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} with better score: {best_score:.4f}")
#                         else:
#                             logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#                     except AttendanceLog.DoesNotExist:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date()
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged for period {period.name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")











# # # (for all active Period) + ATTENDANCE_WINDOW_SECONDS (If new match score is higher → update that log.)
# # ATTENDANCE_WINDOW_SECONDS became a Period model field
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from django.utils.timezone import now
# from datetime import timedelta, date
# import logging
# import os
# import time
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
#
# logger = logging.getLogger(__name__)
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # 🔽 Lowered for better sensitivity
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     attendance_window_seconds = getattr(period, 'attendance_window_seconds', 3600)
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= attendance_window_seconds and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} with better score: {best_score:.4f}")
#                         else:
#                             logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#                     except AttendanceLog.DoesNotExist:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date()
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged for period {period.name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")












# # (for all active Period) + ATTENDANCE_WINDOW_SECONDS (to If yes and new match score is higher → update that log.)
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from django.utils.timezone import now
# from datetime import timedelta, date
# import logging
# import os
# import time
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
#
# logger = logging.getLogger(__name__)
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.45  # 🔽 Lowered for better sensitivity
# ATTENDANCE_WINDOW_SECONDS = 180  # 1 hour window to allow updates
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     try:
#                         log = AttendanceLog.objects.get(
#                             student=best_student,
#                             period=period,
#                             date=current_time.date()
#                         )
#                         time_diff = abs((current_time - log.timestamp).total_seconds())
#
#                         if time_diff <= ATTENDANCE_WINDOW_SECONDS and best_score > log.match_score:
#                             log.match_score = best_score
#                             log.timestamp = current_time
#                             log.camera = camera
#                             log.save()
#                             logger.info(f"🔁 Updated log for {best_student.full_name} with better score: {best_score:.4f}")
#                         else:
#                             logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#                     except AttendanceLog.DoesNotExist:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date()
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} logged for period {period.name} (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")













# #   (for all active Period)
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from django.utils.timezone import now
# from datetime import timedelta, date
# import logging
# import os
# import time
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
#
# logger = logging.getLogger(__name__)
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.45  # 🔽 Lowered for better sensitivity
# ATTENDANCE_WINDOW_SECONDS = 3600  # 1 hour
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 periods = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 )
#
#                 if not periods.exists():
#                     logger.info(f"❌ No active periods at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 for period in periods:
#                     already_logged = AttendanceLog.objects.filter(
#                         student=best_student,
#                         period=period,
#                         date=current_time.date()
#                     ).exists()
#
#                     if not already_logged:
#                         try:
#                             AttendanceLog.objects.create(
#                                 student=best_student,
#                                 period=period,
#                                 camera=camera,
#                                 match_score=best_score,
#                                 timestamp=current_time,
#                                 date=current_time.date()
#                             )
#                             logger.info(f"✅ Match: {best_student.full_name} logged for period {period.name} (Score: {best_score:.4f})")
#                         except Exception as e:
#                             logger.warning(f"❌ Could not log attendance for {best_student.h_code} in period {period.name}: {e}")
#                     else:
#                         logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")







#   (for only the first active Period)
# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from django.utils.timezone import now
# from datetime import timedelta, date
# import logging
# import os
# import time
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
#
# logger = logging.getLogger(__name__)
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.45  # 🔽 Lowered for better sensitivity
# ATTENDANCE_WINDOW_SECONDS = 3600  # 1 hour
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 period = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 ).first()
#
#                 if not period:
#                     logger.info(f"❌ No active period at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 already_logged = AttendanceLog.objects.filter(
#                     student=best_student,
#                     period=period,
#                     date=current_time.date()
#                 ).exists()
#
#                 if not already_logged:
#                     try:
#                         logger.info(f"📝 Logging attendance: {best_student.full_name}, Period: {period.name}, Time: {current_time}, Date: {current_time.date()}")
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time,
#                             date=current_time.date()
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} (Score: {best_score:.4f})")
#                     except Exception as e:
#                         logger.warning(f"❌ Could not log attendance for {best_student.h_code}: {e}")
#                 else:
#                     logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")








# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from django.utils.timezone import now
# from datetime import timedelta, date
# import logging
# import os
# import time
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
#
# logger = logging.getLogger(__name__)
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # 🔽 Lowered for better sensitivity
# ATTENDANCE_WINDOW_SECONDS = 3600  # 1 hour
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 period = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 ).first()
#
#                 if not period:
#                     logger.info(f"❌ No active period at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 already_logged = AttendanceLog.objects.filter(
#                     student=best_student,
#                     period=period,
#                     date=current_time.date()
#                 ).exists()
#
#                 if not already_logged:
#                     try:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=now(),
#                             date=now().date()
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} (Score: {best_score:.4f})")
#                     except Exception as e:
#                         logger.warning(f"❌ Could not log attendance for {best_student.h_code}: {e}")
#                 else:
#                     logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")
#
#
#








# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from django.utils.timezone import now
# from datetime import timedelta, date
# import logging
# import os
# import time
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
#
# logger = logging.getLogger(__name__)
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # 🔽 Lowered for better sensitivity
# ATTENDANCE_WINDOW_SECONDS = 3600  # 1 hour
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 period = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 ).first()
#
#                 if not period:
#                     logger.info(f"❌ No active period at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 already_logged = AttendanceLog.objects.filter(
#                     student=best_student,
#                     period=period,
#                     timestamp__date=current_time.date()
#                 ).exists()
#
#                 if not already_logged:
#                     try:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} (Score: {best_score:.4f})")
#                     except Exception as e:
#                         logger.warning(f"❌ Could not log attendance for {best_student.h_code}: {e}")
#                 else:
#                     logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")













# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from django.utils.timezone import now
# from datetime import timedelta
# import logging
# import os
# import time
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
#
# logger = logging.getLogger(__name__)
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # 🔽 Lowered for better sensitivity
# ATTENDANCE_WINDOW_SECONDS = 3600  # 1 hour
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 except Exception as e:
#                     logger.warning(f"❌ Error calculating cosine similarity: {e}")
#                     continue
#
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 period = Period.objects.filter(is_active=True, start_time__lte=current_time.time(), end_time__gte=current_time.time()).first()
#                 if not period:
#                     logger.info(f"❌ No active period at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 already_logged = AttendanceLog.objects.filter(
#                     student=best_student,
#                     period=period
#                 ).exists()
#
#                 if not already_logged:
#                     try:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} (Score: {best_score:.4f})")
#                     except Exception as e:
#                         logger.warning(f"❌ Could not log attendance for {best_student.h_code}: {e}")
#                 else:
#                     logger.info(f"⏳ Already logged attendance for {best_student.full_name} during this period.")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")
#
#






# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from django.utils.timezone import now
# from datetime import timedelta
# import logging
# import os
# import time
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
#
# logger = logging.getLogger(__name__)
#
# # === 🔧 PARAMETERS ===
# MATCH_THRESHOLD = 0.30  # 🔽 Lowered for better sensitivity
# ATTENDANCE_WINDOW_SECONDS = 3600  # 1 hour
#
# # === 🔁 HELPERS ===
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# # === 🎥 CAMERA STREAM PROCESSING ===
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"⚠️ Failed to read frame from {camera.name}")
#             time.sleep(3)
#             break
#
#         current_time = now()
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ No student found for H-code: {best_h_code}")
#                     continue
#
#                 period = Period.objects.filter(is_active=True, start_time__lte=current_time.time(), end_time__gte=current_time.time()).first()
#                 if not period:
#                     logger.info(f"❌ No active period at {current_time.time()} for {best_student.full_name}")
#                     continue
#
#                 # Check if already logged (just by student + period)
#                 already_logged = AttendanceLog.objects.filter(
#                     student=best_student,
#                     period=period
#                 ).exists()
#
#                 if not already_logged:
#                     try:
#                         AttendanceLog.objects.create(
#                             student=best_student,
#                             period=period,
#                             camera=camera,
#                             match_score=best_score,
#                             timestamp=current_time
#                         )
#                         logger.info(f"✅ Match: {best_student.full_name} (Score: {best_score:.4f})")
#                     except Exception as e:
#                         logger.warning(f"❌ Could not log attendance for {best_student.h_code}: {e}")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🛑 Stopped streaming from camera: {camera.name}")
#















# import cv2
# import numpy as np
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from django.utils.timezone import now
# from datetime import timedelta
# import logging
# import time
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s - %(message)s',
# )
# import os
#
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
#
# logger = logging.getLogger(__name__)
#
# # Thresholds and constants
# MATCH_THRESHOLD = 0.42  # Lower = more strict
# ATTENDANCE_WINDOW_SECONDS = 3600  # 1 hour window
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     if schedule.is_active and weekday in schedule.weekdays:
#         return schedule.start_time <= current_time.time() <= schedule.end_time
#     return False
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for file in os.listdir(embedding_dir):
#         if file.endswith('.npy'):
#             h_code = file.split('.')[0]
#             embedding_path = os.path.join(embedding_dir, file)
#             embedding = np.load(embedding_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
#
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.warning(f"Camera stream not accessible: {camera.name}")
#         return
#
#     logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")
#
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             logger.warning(f"❌ Failed to read frame from {camera.name}. Retrying in 2 seconds...")
#             time.sleep(2)
#             continue
#         else:
#             logger.info(f"📷 Successfully grabbed a frame from {camera.name}")
#
#         current_time = now()
#
#         # ⚠️ Disabled schedule filtering for TESTING mode
#         # if not any(is_within_recognition_schedule(current_time, s) for s in schedules):
#         #     continue
#
#         logger.info(f"📸 Processing frame at {current_time} for {camera.name}")
#
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         if not faces:
#             logger.info("🔍 No faces detected in frame.")
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#
#             for h_code, known_embedding in embeddings_map.items():
#                 score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                 if score > best_score:
#                     best_score = score
#                     best_h_code = h_code
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     best_student = Student.objects.get(h_code=best_h_code)
#                 except Student.DoesNotExist:
#                     logger.warning(f"⚠️ No student found for H-code: {best_h_code}")
#                     continue
#
#                 period = Period.objects.filter(
#                     is_active=True,
#                     start_time__lte=current_time.time(),
#                     end_time__gte=current_time.time()
#                 ).first()
#                 if not period:
#                     logger.info("🕒 No active period found at this time.")
#                     continue
#
#                 time_window_start = current_time - timedelta(seconds=ATTENDANCE_WINDOW_SECONDS)
#                 already_logged = AttendanceLog.objects.filter(
#                     student=best_student,
#                     period=period,
#                     camera=camera,
#                     timestamp__gte=time_window_start
#                 ).exists()
#
#                 if not already_logged:
#                     AttendanceLog.objects.create(
#                         student=best_student,
#                         period=period,
#                         camera=camera,
#                         match_score=best_score,
#                         timestamp=current_time
#                     )
#                     logger.info(f"✅ Match: {best_student.full_name} (Score: {best_score:.4f})")
#                 else:
#                     logger.info(f"🟡 Already logged attendance for {best_student.full_name} recently.")
#             else:
#                 logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")
#
#     cap.release()
#     logger.info(f"🔴 Stopped streaming from camera: {camera.name}")
#
#











#
# # extras/utils.py
# import cv2
# import numpy as np
# from datetime import datetime
# from django.utils.timezone import now
# from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
# from django.utils import timezone
# from insightface.app import FaceAnalysis
# from sklearn.metrics.pairwise import cosine_similarity
# from extras.log_utils import get_camera_logger
#
# IRAQ_TZ = timezone.get_fixed_timezone(180)  # UTC+3
#
# # Check if the current time is within an active recognition schedule
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%a')  # 'Mon', 'Tue', etc.
#     return (
#         schedule.is_active and
#         weekday in schedule.weekdays and
#         schedule.start_time <= current_time.time() <= schedule.end_time
#     )
#
# # Recognize and log attendance from a given camera
#
# # def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
# def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
#     from insightface.app import FaceAnalysis
#     logger = get_camera_logger(camera.name)
#     logger.info(f"Starting stream for camera: {camera.name}")
#
#     # Initialize InsightFace inside the subprocess
#     # face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
#     face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
#
#     logger = get_camera_logger(camera.name)
#     logger.info(f"Starting stream for camera: {camera.name}")
#
#     cap = cv2.VideoCapture(camera.url)
#     if not cap.isOpened():
#         logger.error(f"Failed to open camera stream: {camera.url}")
#         return
#
#     try:
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 logger.warning("Empty frame received")
#                 break
#
#             # Convert BGR to RGB for inference
#             rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             faces = face_analyzer.get(rgb)
#
#             current_time = now().astimezone(IRAQ_TZ)
#
#             for face in faces:
#                 emb = face['embedding'].reshape(1, -1)
#                 best_score = 0
#                 best_student = None
#
#                 for student, known_emb in embeddings_map.items():
#                     sim = cosine_similarity(emb, known_emb.reshape(1, -1))[0][0]
#                     if sim > best_score:
#                         best_score = sim
#                         best_student = student
#
#                 if best_student and best_score > 0.6:  # similarity threshold
#                     logger.info(f"✅ Match: {best_student.full_name} (Score: {best_score:.4f})")
#
#                     for schedule in schedules:
#                         if is_within_recognition_schedule(current_time, schedule):
#                             # Get current period
#                             try:
#                                 current_period = Period.objects.get(
#                                     is_active=True,
#                                     start_time__lte=current_time.time(),
#                                     end_time__gte=current_time.time()
#                                 )
#                             except Period.DoesNotExist:
#                                 logger.warning("No active period found at this time.")
#                                 continue
#
#                             # Save attendance log if not already logged
#                             exists = AttendanceLog.objects.filter(
#                                 student=best_student,
#                                 period=current_period
#                             ).exists()
#
#                             if not exists:
#                                 AttendanceLog.objects.create(
#                                     student=best_student,
#                                     period=current_period,
#                                     camera=camera,
#                                     timestamp=current_time,
#                                     match_score=best_score
#                                 )
#                                 logger.info(f"📝 Attendance logged for {best_student.h_code} in period {current_period.name}")
#                 else:
#                     logger.info(f"❌ No good match (best score: {best_score:.4f})")
#
#     finally:
#         cap.release()
#         logger.info(f"Camera stream closed for: {camera.name}")
#
# import os
# import numpy as np
#
# def load_embeddings(embedding_dir):
#     embeddings_map = {}
#     for filename in os.listdir(embedding_dir):
#         if filename.endswith(".npy"):
#             h_code = os.path.splitext(filename)[0]  # Strip .npy
#             file_path = os.path.join(embedding_dir, filename)
#             embedding = np.load(file_path)
#             embeddings_map[h_code] = embedding
#     return embeddings_map
