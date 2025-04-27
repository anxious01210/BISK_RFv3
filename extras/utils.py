# extras/utils.py
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import logging
import os
import time

from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
from extras.log_utils import get_camera_logger

# === 🔧 PARAMETERS ===
MATCH_THRESHOLD = 0.55  # Lowered for better sensitivity

# === 🔁 HELPERS ===
def is_within_recognition_schedule(current_time, schedule):
    weekday = current_time.strftime('%A')
    return schedule.is_active and weekday in schedule.weekdays and schedule.start_time <= current_time.time() <= schedule.end_time

def load_embeddings(embedding_dir):
    embeddings_map = {}
    for file in os.listdir(embedding_dir):
        if file.endswith('.npy'):
            h_code = file.split('.')[0]
            embedding_path = os.path.join(embedding_dir, file)
            embedding = np.load(embedding_path)
            embeddings_map[h_code] = embedding
    return embeddings_map

# === 🎥 CAMERA STREAM PROCESSING ===
def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
    logger = get_camera_logger(camera.name)  # <-- 🔥 Per camera logger here

    cap = cv2.VideoCapture(camera.url)
    if not cap.isOpened():
        logger.warning(f"⚠️ Camera stream not accessible: {camera.name}")
        return

    logger.info(f"🟢 Streaming from camera: {camera.name} ({camera.location})")

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"⚠️ Failed to read frame from {camera.name}")
            time.sleep(3)
            break

        current_time = datetime.now()
        logger.info(f"📸 Processing frame at {current_time} for {camera.name}")

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_analyzer.get(rgb)

        for face in faces:
            if face.embedding is None:
                continue

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
                            log.save()
                            logger.info(f"🔁 Updated log for {best_student.full_name} with better score: {best_score:.4f}")
                        else:
                            logger.info(f"⏳ Already logged attendance for {best_student.full_name} today for period: {period.name}.")
                    except AttendanceLog.DoesNotExist:
                        AttendanceLog.objects.create(
                            student=best_student,
                            period=period,
                            camera=camera,
                            match_score=best_score,
                            timestamp=current_time,
                            date=current_time.date()
                        )
                        logger.info(f"✅ Match: {best_student.full_name} logged for period {period.name} (Score: {best_score:.4f})")
            else:
                logger.info(f"❌ No match above threshold for detected face (Score: {best_score:.4f})")

    cap.release()
    logger.info(f"🛑 Stopped streaming from camera: {camera.name}")











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
