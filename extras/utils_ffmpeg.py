# # utils_ffmpeg.py
import os
import cv2
import time
import numpy as np
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from django.core.files.base import ContentFile
from django.conf import settings

from attendance.models import AttendanceLog, Student, Period
from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
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

def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
    logger = get_camera_logger(camera.name)
    logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")

    logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
    logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
    logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")

    logger.info(f"🧑‍🎓 Active students: {Student.objects.filter(is_active=True).count()}")
    if LOG_MATCH_THRESHOLD:
        logger.info(f"🔴 MATCH_THRESHOLD = {MATCH_THRESHOLD:.2f}")

    for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        current_time = datetime.now()
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









# import os
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.files.base import ContentFile
# from django.conf import settings
#
# from attendance.models import AttendanceLog, Student, Period
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.50
# RESTART_LIMIT = 5
# RESTART_DELAY = 10  # seconds
#
# LOG_ATTENDANCE = getattr(settings, "LOG_ATTENDANCE", True)
# LOG_MATCHES = getattr(settings, "LOG_MATCHES", True)
# LOG_CROPPED_DEBUG = getattr(settings, "LOG_CROPPED_DEBUG", True)
# LOG_ATTENDANCE_UPDATED = getattr(settings, "LOG_ATTENDANCE_UPDATED", True)
# LOG_MATCH_THRESHOLD = getattr(settings, "LOG_MATCH_THRESHOLD", True)
#
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
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
#         print(f"✅ Loaded {len(embeddings_map)} face embeddings")
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
#
# def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     logger.info(f"🧑‍🎓 Active students: {Student.objects.filter(is_active=True).count()}")
#     if LOG_MATCH_THRESHOLD:
#         logger.info(f"🔴 MATCH_THRESHOLD = {MATCH_THRESHOLD:.2f}")
#
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         current_time = datetime.now()
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     known_norm = known_embedding / np.linalg.norm(known_embedding)
#                     face_norm = face.embedding / np.linalg.norm(face.embedding)
#                     score = cosine_similarity([face_norm], [known_norm])[0][0]
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Similarity error with {h_code}: {e}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                     display_name = f"{student.h_code}__{student.full_name}"
#                     if LOG_MATCHES:
#                         logger.info(f"🧬 Best match {display_name} | Score: {best_score:.4f}")
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ Student not found for H-code: {best_h_code}")
#                     continue
#
#                 active_periods = Period.objects.filter(is_active=True)
#                 for period in active_periods:
#                     if period.start_time <= current_time.time() <= period.end_time:
#                         log, created = AttendanceLog.objects.get_or_create(
#                             student=student,
#                             period=period,
#                             date=current_time.date(),
#                             defaults={
#                                 "camera": camera,
#                                 "timestamp": current_time,
#                                 "match_score": best_score
#                             }
#                         )
#
#                         updated = False
#                         if not created:
#                             if best_score > log.match_score:
#                                 log.timestamp = current_time
#                                 log.match_score = best_score
#                                 log.camera = camera
#                                 updated = True
#                                 if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                                     log.cropped_face.delete(save=False)
#
#                         if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 x1, y1, x2, y2 = map(int, face.bbox)
#                                 cropped_face = rgb[y1:y2, x1:x2]
#                                 _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                                 log.cropped_face.save(
#                                     get_attendance_crop_path(student.h_code, camera.name),
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#                                 updated = True
#                             except Exception as e:
#                                 logger.error(f"❌ Saving cropped image failed: {e}")
#
#                         if created:
#                             log.save()
#                             if LOG_ATTENDANCE:
#                                 logger.info(f"✅ Attendance saved for {display_name} in period '{period.name}' (Score: {best_score:.4f})")
#                         elif updated:
#                             log.save()
#                             if LOG_ATTENDANCE_UPDATED:
#                                 logger.info(f"🟠 Attendance updated for {display_name} in period '{period.name}' (New Score: {best_score:.4f})")
#                         else:
#                             if LOG_ATTENDANCE_UPDATED:
#                                 logger.info(f"🟡 Attendance already exists with equal or better score for {display_name} in period '{period.name}'")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#             if getattr(settings, "SAVE_ALL_CROPPED_FACES", True) and LOG_CROPPED_DEBUG:
#                 try:
#                     x1, y1, x2, y2 = map(int, face.bbox)
#                     cropped_face = rgb[y1:y2, x1:x2]
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")










# import os
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.files.base import ContentFile
# from django.conf import settings
#
# from attendance.models import AttendanceLog, Student, Period
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.50
# RESTART_LIMIT = 5
# RESTART_DELAY = 10  # seconds
#
# LOG_ATTENDANCE = getattr(settings, "LOG_ATTENDANCE", True)
# LOG_MATCHES = getattr(settings, "LOG_MATCHES", True)
# LOG_CROPPED_DEBUG = getattr(settings, "LOG_CROPPED_DEBUG", True)
# LOG_ATTENDANCE_UPDATED = getattr(settings, "LOG_ATTENDANCE_UPDATED", True)
# LOG_MATCH_THRESHOLD = getattr(settings, "LOG_MATCH_THRESHOLD", True)
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
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
#         print(f"✅ Loaded {len(embeddings_map)} face embeddings")
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
# def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     logger.info(f"🧑‍🎓 Active students: {Student.objects.filter(is_active=True).count()}")
#     if LOG_MATCH_THRESHOLD:
#         logger.info(f"🔴 MATCH_THRESHOLD = {MATCH_THRESHOLD:.2f}")
#
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         current_time = datetime.now()
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     known_norm = known_embedding / np.linalg.norm(known_embedding)
#                     face_norm = face.embedding / np.linalg.norm(face.embedding)
#                     score = cosine_similarity([face_norm], [known_norm])[0][0]
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Similarity error with {h_code}: {e}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                     if LOG_MATCHES:
#                         logger.info(f"🧬 Best match H-code: {best_h_code} | Score: {best_score:.4f}")
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ Student not found for H-code: {best_h_code}")
#                     continue
#
#                 active_periods = Period.objects.filter(is_active=True)
#                 for period in active_periods:
#                     if period.start_time <= current_time.time() <= period.end_time:
#                         log, created = AttendanceLog.objects.get_or_create(
#                             student=student,
#                             period=period,
#                             date=current_time.date(),
#                             defaults={
#                                 "camera": camera,
#                                 "timestamp": current_time,
#                                 "match_score": best_score
#                             }
#                         )
#
#                         updated = False
#                         if not created:
#                             if best_score > log.match_score:
#                                 log.timestamp = current_time
#                                 log.match_score = best_score
#                                 log.camera = camera
#                                 updated = True
#                                 if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                                     log.cropped_face.delete(save=False)
#
#                         if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 x1, y1, x2, y2 = map(int, face.bbox)
#                                 cropped_face = rgb[y1:y2, x1:x2]
#                                 _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                                 log.cropped_face.save(
#                                     get_attendance_crop_path(student.h_code, camera.name),
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#                                 updated = True
#                             except Exception as e:
#                                 logger.error(f"❌ Saving cropped image failed: {e}")
#
#                         if created:
#                             log.save()
#                             if LOG_ATTENDANCE:
#                                 logger.info(f"✅ Attendance saved for {student.full_name} in period '{period.name}' (Score: {best_score:.4f})")
#                         elif updated:
#                             log.save()
#                             if LOG_ATTENDANCE_UPDATED:
#                                 logger.info(f"🟠 Attendance updated for {student.full_name} in period '{period.name}' (New Score: {best_score:.4f})")
#                         else:
#                             if LOG_ATTENDANCE:
#                                 logger.debug(f"🟡 Attendance already exists with equal or better score for {student.full_name} in period '{period.name}'")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#             if getattr(settings, "SAVE_ALL_CROPPED_FACES", True) and LOG_CROPPED_DEBUG:
#                 try:
#                     x1, y1, x2, y2 = map(int, face.bbox)
#                     cropped_face = rgb[y1:y2, x1:x2]
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")






# import os
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.files.base import ContentFile
# from django.conf import settings
#
# from attendance.models import AttendanceLog, Student, Period
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.50
# RESTART_LIMIT = 5
# RESTART_DELAY = 10  # seconds
#
# LOG_ATTENDANCE = getattr(settings, "LOG_ATTENDANCE", True)
# LOG_MATCHES = getattr(settings, "LOG_MATCHES", True)
# LOG_CROPPED_DEBUG = getattr(settings, "LOG_CROPPED_DEBUG", True)
# LOG_ATTENDANCE_UPDATED = getattr(settings, "LOG_ATTENDANCE_UPDATED", True)
# LOG_MATCH_THRESHOLD = getattr(settings, "LOG_MATCH_THRESHOLD", True)
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
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
#         print(f"✅ Loaded {len(embeddings_map)} face embeddings")
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
# def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     logger.info(f"🧑‍🎓 Active students: {Student.objects.filter(is_active=True).count()}")
#     if LOG_MATCH_THRESHOLD:
#         logger.info(f"🔴 MATCH_THRESHOLD = {MATCH_THRESHOLD:.2f}")
#
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         current_time = datetime.now()
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     known_norm = known_embedding / np.linalg.norm(known_embedding)
#                     face_norm = face.embedding / np.linalg.norm(face.embedding)
#                     score = cosine_similarity([face_norm], [known_norm])[0][0]
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Similarity error with {h_code}: {e}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                     if LOG_MATCHES:
#                         logger.info(f"🧬 Best match H-code: {best_h_code}_{student.full_name} | Score: {best_score:.4f}")
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ Student not found for H-code: {best_h_code}")
#                     continue
#
#                 active_periods = Period.objects.filter(is_active=True)
#                 for period in active_periods:
#                     if period.start_time <= current_time.time() <= period.end_time:
#                         log, created = AttendanceLog.objects.get_or_create(
#                             student=student,
#                             period=period,
#                             date=current_time.date(),
#                             defaults={
#                                 "camera": camera,
#                                 "timestamp": current_time,
#                                 "match_score": best_score
#                             }
#                         )
#
#                         updated = False
#                         if not created:
#                             if best_score > log.match_score:
#                                 log.timestamp = current_time
#                                 log.match_score = best_score
#                                 log.camera = camera
#                                 updated = True
#                                 if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                                     log.cropped_face.delete(save=False)
#
#                         if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 x1, y1, x2, y2 = map(int, face.bbox)
#                                 cropped_face = rgb[y1:y2, x1:x2]
#                                 _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                                 log.cropped_face.save(
#                                     get_attendance_crop_path(student.h_code, camera.name),
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#                                 updated = True
#                             except Exception as e:
#                                 logger.error(f"❌ Saving cropped image failed: {e}")
#
#                         if created:
#                             log.save()
#                             if LOG_ATTENDANCE:
#                                 logger.info(f"✅ Attendance saved for {student.full_name} in period '{period.name}' (Score: {best_score:.4f})")
#                         elif updated:
#                             log.save()
#                             if LOG_ATTENDANCE_UPDATED:
#                                 logger.info(f"🟠 Attendance updated for {student.full_name} in period '{period.name}' (New Score: {best_score:.4f})")
#                         else:
#                             if LOG_ATTENDANCE:
#                                 logger.debug(f"🟡 Attendance already exists with equal or better score for {student.full_name} in period '{period.name}'")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#             if getattr(settings, "SAVE_ALL_CROPPED_FACES", True) and LOG_CROPPED_DEBUG:
#                 try:
#                     x1, y1, x2, y2 = map(int, face.bbox)
#                     cropped_face = rgb[y1:y2, x1:x2]
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")









# # utils_ffmpeg.py
# import os
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.files.base import ContentFile
# from django.conf import settings
#
# from attendance.models import AttendanceLog, Student, Period
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.50
# RESTART_LIMIT = 5
# RESTART_DELAY = 10  # seconds
#
# LOG_ATTENDANCE = getattr(settings, "LOG_ATTENDANCE", True)
# LOG_MATCHES = getattr(settings, "LOG_MATCHES", True)
# LOG_CROPPED_DEBUG = getattr(settings, "LOG_CROPPED_DEBUG", True)
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
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
#         print(f"✅ Loaded {len(embeddings_map)} face embeddings")
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
# def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     logger.info(f"🧑‍🎓 Active students: {Student.objects.filter(is_active=True).count()}")
#
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         current_time = datetime.now()
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     known_norm = known_embedding / np.linalg.norm(known_embedding)
#                     face_norm = face.embedding / np.linalg.norm(face.embedding)
#                     score = cosine_similarity([face_norm], [known_norm])[0][0]
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Similarity error with {h_code}: {e}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                     if LOG_MATCHES:
#                         logger.info(f"🧬 Best match H-code: {best_h_code} | Score: {best_score:.4f}")
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ Student not found for H-code: {best_h_code}")
#                     continue
#
#                 active_periods = Period.objects.filter(is_active=True)
#                 for period in active_periods:
#                     if period.start_time <= current_time.time() <= period.end_time:
#                         log, created = AttendanceLog.objects.get_or_create(
#                             student=student,
#                             period=period,
#                             date=current_time.date(),
#                             defaults={
#                                 "camera": camera,
#                                 "timestamp": current_time,
#                                 "match_score": best_score
#                             }
#                         )
#
#                         updated = False
#                         if not created:
#                             if best_score > log.match_score:
#                                 log.timestamp = current_time
#                                 log.match_score = best_score
#                                 log.camera = camera
#                                 updated = True
#                                 if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                                     log.cropped_face.delete(save=False)
#                             else:
#                                 continue  # skip if not better
#
#                         if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 x1, y1, x2, y2 = map(int, face.bbox)
#                                 cropped_face = rgb[y1:y2, x1:x2]
#                                 _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                                 log.cropped_face.save(
#                                     get_attendance_crop_path(student.h_code, camera.name),
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#                                 updated = True
#                             except Exception as e:
#                                 logger.error(f"❌ Saving cropped image failed: {e}")
#
#                         if created or updated:
#                             log.save()
#                             if LOG_ATTENDANCE:
#                                 logger.info(f"✅ Attendance saved for {student.full_name} in period '{period.name}' (Score: {best_score:.4f})")
#                         else:
#                             if LOG_ATTENDANCE:
#                                 logger.debug(f"🟡 Attendance already exists with equal or better score for {student.full_name} in period '{period.name}'")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#             if getattr(settings, "SAVE_ALL_CROPPED_FACES", True) and LOG_CROPPED_DEBUG:
#                 try:
#                     x1, y1, x2, y2 = map(int, face.bbox)
#                     cropped_face = rgb[y1:y2, x1:x2]
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")













# # utils_ffmpeg.py
# import os
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.files.base import ContentFile
# from django.conf import settings
#
# from attendance.models import AttendanceLog, Student, Period
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.50
# RESTART_LIMIT = 5
# RESTART_DELAY = 10  # seconds
#
# LOG_ATTENDANCE = getattr(settings, "LOG_ATTENDANCE", True)
# LOG_MATCHES = getattr(settings, "LOG_MATCHES", True)
# LOG_CROPPED_DEBUG = getattr(settings, "LOG_CROPPED_DEBUG", True)
#
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
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
#         print(f"✅ Loaded {len(embeddings_map)} face embeddings")
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
#
# def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     logger.info(f"🧑‍🎓 Active students: {Student.objects.filter(is_active=True).count()}")
#
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         current_time = datetime.now()
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     known_norm = known_embedding / np.linalg.norm(known_embedding)
#                     face_norm = face.embedding / np.linalg.norm(face.embedding)
#                     score = cosine_similarity([face_norm], [known_norm])[0][0]
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Similarity error with {h_code}: {e}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                     if LOG_MATCHES:
#                         logger.info(f"🧬 Best match H-code: {best_h_code} | Score: {best_score:.4f}")
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ Student not found for H-code: {best_h_code}")
#                     continue
#
#                 active_periods = Period.objects.filter(is_active=True)
#                 for period in active_periods:
#                     if period.start_time <= current_time.time() <= period.end_time:
#                         log, created = AttendanceLog.objects.get_or_create(
#                             student=student,
#                             period=period,
#                             date=current_time.date(),
#                             defaults={
#                                 "camera": camera,
#                                 "timestamp": current_time,
#                                 "match_score": best_score
#                             }
#                         )
#
#                         updated = False
#                         if not created and best_score > log.match_score:
#                             log.timestamp = current_time
#                             log.match_score = best_score
#                             log.camera = camera
#                             updated = True
#                             if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                                 log.cropped_face.delete(save=False)
#
#                         if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 x1, y1, x2, y2 = map(int, face.bbox)
#                                 cropped_face = rgb[y1:y2, x1:x2]
#                                 _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                                 log.cropped_face.save(
#                                     get_attendance_crop_path(student.h_code, camera.name),
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#                                 updated = True
#                             except Exception as e:
#                                 logger.error(f"❌ Saving cropped image failed: {e}")
#                         if updated or created:
#                             log.save()
#                             if LOG_ATTENDANCE:
#                                 logger.info(f"✅ Attendance saved for {student.full_name} in period '{period.name}' (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#             if getattr(settings, "SAVE_ALL_CROPPED_FACES", True) and LOG_CROPPED_DEBUG:
#                 try:
#                     x1, y1, x2, y2 = map(int, face.bbox)
#                     cropped_face = rgb[y1:y2, x1:x2]
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")








# # utils.py working FINE
# import os
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.files.base import ContentFile
# from django.conf import settings
#
# from attendance.models import AttendanceLog, Student, Period
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.50
# RESTART_LIMIT = 5
# RESTART_DELAY = 10  # seconds
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return (
#         schedule.is_active and
#         weekday in schedule.weekdays and
#         schedule.start_time <= current_time.time() <= schedule.end_time
#     )
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
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
#         print(f"✅ Loaded {len(embeddings_map)} face embeddings")
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
# def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     logger.info(f"🧑‍🎓 Active students: {Student.objects.filter(is_active=True).count()}")
#
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         current_time = datetime.now()
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#             score_list = []
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     known_norm = known_embedding / np.linalg.norm(known_embedding)
#                     face_norm = face.embedding / np.linalg.norm(face.embedding)
#                     score = cosine_similarity([face_norm], [known_norm])[0][0]
#                     score_list.append((h_code, score))
#                     logger.debug(f"▶️ Comparing with {h_code} | Score: {score:.4f}")
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Similarity error with {h_code}: {e}")
#
#             logger.debug(f"↩️ Best match: {best_h_code} | Score: {best_score:.4f}")
#             for top_h, top_score in sorted(score_list, key=lambda x: x[1], reverse=True)[:3]:
#                 logger.debug(f"🔍 Top match candidate: {top_h} | Score: {top_score:.4f}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                     logger.info(f"🧬 Best match H-code: {best_h_code} | Score: {best_score:.4f}")
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ Student not found for H-code: {best_h_code}")
#                     continue
#
#                 active_periods = Period.objects.filter(is_active=True)
#                 for period in active_periods:
#                     if period.start_time <= current_time.time() <= period.end_time:
#                         log, created = AttendanceLog.objects.get_or_create(
#                             student=student,
#                             period=period,
#                             date=current_time.date(),
#                             defaults={
#                                 "camera": camera,
#                                 "timestamp": current_time,
#                                 "match_score": best_score
#                             }
#                         )
#
#                         if not created and best_score > log.match_score:
#                             log.timestamp = current_time
#                             log.match_score = best_score
#                             log.camera = camera
#                             if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                                 log.cropped_face.delete(save=False)
#
#                         if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 x1, y1, x2, y2 = map(int, face.bbox)
#                                 cropped_face = rgb[y1:y2, x1:x2]
#                                 _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                                 log.cropped_face.save(
#                                     get_attendance_crop_path(student.h_code, camera.name),
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#                             except Exception as e:
#                                 logger.error(f"❌ Saving cropped image failed: {e}")
#                         log.save()
#                         logger.info(f"✅ Attendance saved for {student.full_name} in period '{period.name}' (Score: {best_score:.4f})")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#             if getattr(settings, "SAVE_ALL_CROPPED_FACES", True):
#                 try:
#                     x1, y1, x2, y2 = map(int, face.bbox)
#                     cropped_face = rgb[y1:y2, x1:x2]
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")









# import os
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.files.base import ContentFile
# from django.conf import settings
#
# from attendance.models import AttendanceLog, Student, Period
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.50
# RESTART_LIMIT = 5
# RESTART_DELAY = 10  # seconds
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return (
#         schedule.is_active and
#         weekday in schedule.weekdays and
#         schedule.start_time <= current_time.time() <= schedule.end_time
#     )
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
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
#         print(f"✅ Loaded {len(embeddings_map)} face embeddings")
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
# def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     logger.info(f"🧑‍🎓 Active students: {Student.objects.filter(is_active=True).count()}")
#
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         current_time = datetime.now()
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
#             score_list = []
#
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     known_norm = known_embedding / np.linalg.norm(known_embedding)
#                     face_norm = face.embedding / np.linalg.norm(face.embedding)
#                     score = cosine_similarity([face_norm], [known_norm])[0][0]
#                     score_list.append((h_code, score))
#                     logger.debug(f"▶️ Comparing with {h_code} | Score: {score:.4f}")
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Similarity error with {h_code}: {e}")
#
#             logger.debug(f"↩️ Best match: {best_h_code} | Score: {best_score:.4f}")
#             for top_h, top_score in sorted(score_list, key=lambda x: x[1], reverse=True)[:3]:
#                 logger.debug(f"🔍 Top match candidate: {top_h} | Score: {top_score:.4f}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     logger.info(f"🧬 Best match H-code: {best_h_code} | Score: {best_score:.4f}")
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ Student not found for H-code: {best_h_code}")
#                     continue
#
#                 for sched in schedules:
#                     if not is_within_recognition_schedule(current_time, sched):
#                         logger.debug(f"📅 Skipping schedule '{sched}' - Outside active window or wrong weekday")
#                         continue
#
#                     period = sched.period
#                     log, created = AttendanceLog.objects.get_or_create(
#                         student=student,
#                         period=period,
#                         date=current_time.date(),
#                         defaults={
#                             "camera": camera,
#                             "timestamp": current_time,
#                             "match_score": best_score
#                         }
#                     )
#
#                     if not created and best_score > log.match_score:
#                         log.timestamp = current_time
#                         log.match_score = best_score
#                         log.camera = camera
#                         if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                             log.cropped_face.delete(save=False)
#
#                     if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                         try:
#                             x1, y1, x2, y2 = map(int, face.bbox)
#                             cropped_face = rgb[y1:y2, x1:x2]
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
#
#                 logger.info(f"✅ Schedule matched: {sched.name} | Period: {sched.period.name}")
#             else:
#                 logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#             if getattr(settings, "SAVE_ALL_CROPPED_FACES", True):
#                 try:
#                     x1, y1, x2, y2 = map(int, face.bbox)
#                     cropped_face = rgb[y1:y2, x1:x2]
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")











# import os
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.files.base import ContentFile
# from django.conf import settings
#
# from attendance.models import AttendanceLog, Student
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.25
# RESTART_LIMIT = 5
# RESTART_DELAY = 10  # seconds
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return (
#         schedule.is_active and
#         weekday in schedule.weekdays and
#         schedule.start_time <= current_time.time() <= schedule.end_time
#     )
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
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
#         print(f"✅ Loaded {len(embeddings_map)} face embeddings")
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
# def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     logger.info(f"🧑‍🎓 Active students: {Student.objects.filter(is_active=True).count()}")
#
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         current_time = datetime.now()
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
#                     known_norm = known_embedding / np.linalg.norm(known_embedding)
#                     face_norm = face.embedding / np.linalg.norm(face.embedding)
#
#                     score = cosine_similarity([face_norm], [known_norm])[0][0]
#                     logger.debug(f"▶️ Comparing with {h_code} | Score: {score:.4f}")
#
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                 except Exception as e:
#                     logger.warning(f"Similarity error with {h_code}: {e}")
#
#             logger.debug(f"↩️ Best match: {best_h_code} | Score: {best_score:.4f}")
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
#                         logger.debug(f"📅 Skipping schedule '{sched}' - Outside active window or wrong weekday")
#                         continue
#
#                     period = sched.period
#                     log, created = AttendanceLog.objects.get_or_create(
#                         student=student,
#                         period=period,
#                         date=current_time.date(),
#                         defaults={
#                             "camera": camera,
#                             "timestamp": current_time,
#                             "match_score": best_score
#                         }
#                     )
#
#                     if not created and best_score > log.match_score:
#                         log.timestamp = current_time
#                         log.match_score = best_score
#                         log.camera = camera
#                         if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                             log.cropped_face.delete(save=False)
#
#                     if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                         try:
#                             x1, y1, x2, y2 = map(int, face.bbox)
#                             cropped_face = rgb[y1:y2, x1:x2]
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
#                     x1, y1, x2, y2 = map(int, face.bbox)
#                     cropped_face = rgb[y1:y2, x1:x2]
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")




# # utils_ffmpeg.py
# import os
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.files.base import ContentFile
# from django.conf import settings
#
# from attendance.models import AttendanceLog, Student
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.50
# RESTART_LIMIT = 5
# RESTART_DELAY = 10  # seconds
#
# def is_within_recognition_schedule(current_time, schedule):
#     weekday = current_time.strftime('%A')
#     return (
#         schedule.is_active and
#         weekday in schedule.weekdays and
#         schedule.start_time <= current_time.time() <= schedule.end_time
#     )
#
# def get_attendance_crop_path(student_h_code, camera_name):
#     today = datetime.now()
#     return f"attendance_crops/{today.strftime('%Y/%m/%d')}/{student_h_code}_{camera_name}_{today.strftime('%Y%m%d_%H%M%S')}.jpg"
#
# def load_embeddings(embedding_dir):
#     import pickle
#     embeddings_map = {}
#     npy_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
#     if npy_files:
#         for file in npy_files:
#             h_code = file.split('.')[0]
#             embedding = np.load(os.path.join(embedding_dir, file))
#             embedding = embedding / np.linalg.norm(embedding)
#             embeddings_map[h_code] = embedding
#     else:
#         fallback = os.path.join(settings.MEDIA_ROOT, "face_embeddings.pkl")
#         if os.path.exists(fallback):
#             with open(fallback, "rb") as f:
#                 embeddings_map = pickle.load(f)
#             print("⚠️ Loaded fallback face_embeddings.pkl")
#         else:
#             print("❌ No embeddings found")
#     print(f"✅ Loaded {len(embeddings_map)} face embeddings")
#     return embeddings_map
#
# def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         current_time = datetime.now()
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 face.embedding = face.embedding / np.linalg.norm(face.embedding)
#                 logger.debug("❌ Face has no embedding, skipping...")
#                 continue
#
#             best_score = -1
#             best_h_code = None
#             print(f"🧑‍🎓 Active students: {Student.objects.filter(is_active=True).count()}")
#             for h_code, known_embedding in embeddings_map.items():
#                 try:
#                     score = cosine_similarity([face.embedding], [known_embedding])[0][0]
#                     if score > best_score:
#                         best_score = score
#                         best_h_code = h_code
#                     else:
#                         logger.info(f"❌ No match above threshold (Score: {best_score:.4f})")
#
#                 except Exception as e:
#                     logger.warning(f"Similarity error: {e}")
#
#             logger.debug(f"↩️ Match candidate: {best_h_code} | Score: {best_score:.4f}")
#
#             if best_score >= MATCH_THRESHOLD and best_h_code:
#                 try:
#                     logger.debug(f"🔍 Trying to fetch Student with h_code={best_h_code}")
#                     student = Student.objects.get(h_code=best_h_code, is_active=True)
#                 except Student.DoesNotExist:
#                     logger.warning(f"❌ Student not found for H-code: {best_h_code}")
#                     continue
#
#                 for sched in schedules:
#                     if not is_within_recognition_schedule(current_time, sched):
#                         logger.debug(f"📅 Skipping schedule '{sched}' - Outside active window or wrong weekday")
#                         continue
#
#                     period = sched.period
#                     log, created = AttendanceLog.objects.get_or_create(
#                         student=student,
#                         period=period,
#                         date=current_time.date(),
#                         defaults={
#                             "camera": camera,
#                             "timestamp": current_time,
#                             "match_score": best_score
#                         }
#                     )
#
#                     if not created and best_score > log.match_score:
#                         log.timestamp = current_time
#                         log.match_score = best_score
#                         log.camera = camera
#                         if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                             log.cropped_face.delete(save=False)
#
#                     if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                         try:
#                             x1, y1, x2, y2 = map(int, face.bbox)
#                             cropped_face = rgb[y1:y2, x1:x2]
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
#                     x1, y1, x2, y2 = map(int, face.bbox)
#                     cropped_face = rgb[y1:y2, x1:x2]
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")







# # extras/utils_ffmpeg.py
# import os
# import cv2
# import time
# import numpy as np
# from datetime import datetime
# from sklearn.metrics.pairwise import cosine_similarity
# from django.core.files.base import ContentFile
# from django.conf import settings
#
# from attendance.models import AttendanceLog, Student
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
# from extras.log_utils import get_camera_logger
#
# MATCH_THRESHOLD = 0.50
# RESTART_LIMIT = 5
# RESTART_DELAY = 10  # seconds
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
# def process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map):
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🟢 FFmpeg stream started for camera: {camera.name}")
#
#     logger.info(f"SAVE_ALL_CROPPED_FACES = {getattr(settings, 'SAVE_ALL_CROPPED_FACES', True)}")
#     logger.info(f"SAVE_CROPPED_IMAGE = {getattr(settings, 'SAVE_CROPPED_IMAGE', True)}")
#     logger.info(f"DELETE_OLD_CROPPED_IMAGE = {getattr(settings, 'DELETE_OLD_CROPPED_IMAGE', False)}")
#
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         current_time = datetime.now()
#         faces = face_analyzer.get(rgb)
#
#         for face in faces:
#             if face.embedding is None:
#                 continue
#
#             best_score = -1
#             best_h_code = None
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
#                     if sched.start_time <= current_time.time() <= sched.end_time:
#                         period = sched.period
#                         log, created = AttendanceLog.objects.get_or_create(
#                             student=student,
#                             period=period,
#                             date=current_time.date(),
#                             defaults={"camera": camera, "timestamp": current_time, "match_score": best_score}
#                         )
#                         if not created and best_score > log.match_score:
#                             log.timestamp = current_time
#                             log.match_score = best_score
#                             log.camera = camera
#                             if getattr(settings, "DELETE_OLD_CROPPED_IMAGE", False) and log.cropped_face:
#                                 log.cropped_face.delete(save=False)
#
#                         if not log.cropped_face and getattr(settings, "SAVE_CROPPED_IMAGE", True):
#                             try:
#                                 x1, y1, x2, y2 = map(int, face.bbox)
#                                 cropped_face = rgb[y1:y2, x1:x2]
#                                 _, img_encoded = cv2.imencode(".jpg", cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                                 log.cropped_face.save(
#                                     f"attendance_crops/{current_time.strftime('%Y/%m/%d')}/{student.h_code}_{camera.name}_{current_time.strftime('%H%M%S')}.jpg",
#                                     ContentFile(img_encoded.tobytes()),
#                                     save=False
#                                 )
#                             except Exception as e:
#                                 logger.error(f"❌ Saving cropped image failed: {e}")
#                         log.save()
#                         logger.info(f"✅ Attendance saved for {student.full_name} (Score: {best_score:.4f})")
#                     else:
#                         logger.debug("⌛ Outside schedule, skipping logging")
#
#             if getattr(settings, "SAVE_ALL_CROPPED_FACES", True):
#                 try:
#                     x1, y1, x2, y2 = map(int, face.bbox)
#                     cropped_face = rgb[y1:y2, x1:x2]
#                     debug_dir = os.path.join(settings.MEDIA_ROOT, "logs", "debug_faces", camera.name)
#                     os.makedirs(debug_dir, exist_ok=True)
#                     filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
#                     debug_path = os.path.join(debug_dir, filename)
#                     cv2.imwrite(debug_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
#                     logger.info(f"🧪 Debug cropped face saved: {debug_path}")
#                 except Exception as e:
#                     logger.warning(f"⚠️ Debug save failed: {e}")







# # extras/utils_ffmpeg.py
# import os
# import cv2
# import numpy as np
# import time
# from datetime import datetime
# from django.conf import settings
# from django.core.files.base import ContentFile
# from insightface.utils.face_align import norm_crop
# from sklearn.metrics.pairwise import cosine_similarity
#
# from attendance.models import AttendanceLog, Student
# from extras.log_utils import get_camera_logger
# from extras.stream_utils_ffmpeg import stream_frames_ffmpeg
#
# MATCH_THRESHOLD = 0.50
# RESTART_LIMIT = 5
# RESTART_DELAY = 10
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
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🎥 [START] Processing stream for: {camera.name}")
#
#     logger.info(f"Using FFmpeg stream for {camera.name}")
#
#     use_alignment = hasattr(face_analyzer, 'align_crop')
#     if use_alignment:
#         logger.info("✅ Face alignment (align_crop) is ENABLED.")
#     else:
#         logger.warning("⚠️ Face alignment (align_crop) is NOT available. Running without alignment.")
#
#     read_failure_count = 0
#     for frame in stream_frames_ffmpeg(camera.url, fps_limit=1):
#         if frame is None or frame.size == 0:
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
#     logger.info(f"🚩 Stopped FFmpeg stream from camera: {camera.name}")
