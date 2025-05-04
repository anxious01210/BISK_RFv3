# extras/recognize_and_log_attendance_parallel.py (Here InsightFace handles defining the det_set Automatically)
import os
import sys
import django
import logging
from datetime import datetime
import multiprocessing

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
django.setup()

from attendance.models import Camera, RecognitionSchedule
# 👇 These are shared logic/utilities from utils.py
from extras.utils import process_camera_stream, is_within_recognition_schedule, load_embeddings, RESTART_LIMIT, RESTART_DELAY

def run_camera(camera, schedules, embedding_dir):
    from insightface.app import FaceAnalysis  # avoid GPU context collision
    from extras.log_utils import get_camera_logger  # <-- Move import here too
    import time

    logger = get_camera_logger(camera.name)
    logger.info(f"🎥 [START] Processing stream for: {camera.name}")

    # Initialize FaceAnalyzer without explicitly setting the det_size for automatic resizing
    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # If you want slightly better accuracy for very small faces, you could increase it to 768×768 or 800×800, but this will slow down processing.
    # 640×640 is a good balance (speed vs accuracy).
    # face_analyzer.prepare(ctx_id=0, det_size=(800, 800))
    # face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))
    # face_analyzer.prepare(ctx_id=0, det_size=(1600, 1600))
    # face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))
    # face_analyzer.prepare(ctx_id=0, det_size=(2144, 2144))

    # ✅ Allow the model to handle resizing automatically instead of specifying det_size
    face_analyzer.prepare(ctx_id=0)

    embeddings_map = load_embeddings(embedding_dir)

    # === 🛠️ Retry mechanism here ===
    # The retry logic is placed here in the parallel script (not utils.py) because it's process-level logic.
    # We keep constants like RESTART_LIMIT and RESTART_DELAY in utils.py to centralize config, but retry should
    # happen here where camera multiprocessing context is handled.
    attempts = 0
    while attempts < RESTART_LIMIT:
        try:
            logger.info(f"🚀 Attempt {attempts+1}/{RESTART_LIMIT} for camera: {camera.name}")
            process_camera_stream(camera, schedules, face_analyzer, embeddings_map)
            break  # ✅ Exit loop if processing completes without exception
        except Exception as e:
            logger.exception(f"🔥 [ERROR] Exception while processing camera {camera.name}: {e}")
            attempts += 1
            if attempts < RESTART_LIMIT:
                logger.info(f"🔁 Retry {attempts}/{RESTART_LIMIT} after {RESTART_DELAY} seconds")
                time.sleep(RESTART_DELAY)
            else:
                logger.error(f"🛑 Max retries reached for camera {camera.name}. Giving up.")

def recognize_and_log():
    print("🔧 Loading active cameras and recognition schedules...")

    embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")

    now = datetime.now()
    today_weekday = now.strftime('%a')[:3]  # e.g., 'Mon', 'Tue'

    active_cameras = []
    camera_schedules_map = {}

    for camera in Camera.objects.filter(is_active=True):
        schedules = RecognitionSchedule.objects.filter(is_active=True, cameras=camera)

        # Filter for valid current schedules
        valid_schedules = [
            s for s in schedules
            if today_weekday in s.weekdays and s.start_time <= now.time() <= s.end_time
        ]

        if valid_schedules:
            active_cameras.append(camera)
            camera_schedules_map[camera.id] = valid_schedules

    processes = []
    for camera in active_cameras:
        print(f"🚀 Spawning process for camera: {camera.name}")
        p = multiprocessing.Process(target=run_camera, args=(camera, camera_schedules_map[camera.id], embedding_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("✅ All camera processes have completed.")

if __name__ == "__main__":
    print("🚀 Starting recognition and log attendance (parallel)...")
    recognize_and_log()



















# # extras/recognize_and_log_attendance_parallel.py (Here InsightFace handles defining the det_set Automatically)
# import os
# import sys
# import django
# import logging
# from datetime import datetime
# import multiprocessing
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
# django.setup()
#
# from attendance.models import Camera, RecognitionSchedule
# from extras.utils import process_camera_stream, is_within_recognition_schedule, load_embeddings, RESTART_LIMIT, RESTART_DELAY
#
# def run_camera(camera, schedules, embedding_dir):
#     from insightface.app import FaceAnalysis  # avoid GPU context collision
#     from extras.log_utils import get_camera_logger  # <-- Move import here too
#     import time
#
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🎥 [START] Processing stream for: {camera.name}")
#
#     # Initialize FaceAnalyzer without explicitly setting the det_size for automatic resizing
#     face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#
#     # f you want slightly better accuracy for very small faces, you could increase it to 768×768 or 800×800, but this will slow down processing.
#     # 640×640 is a good balance (speed vs accuracy).
#     # face_analyzer.prepare(ctx_id=0, det_size=(800, 800))
#     # face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))
#     # face_analyzer.prepare(ctx_id=0, det_size=(1600, 1600))
#     # face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))
#     # face_analyzer.prepare(ctx_id=0, det_size=(2144, 2144))
#
#     # Allow the model to handle resizing automatically instead of specifying det_size
#     face_analyzer.prepare(ctx_id=0)
#
#     embeddings_map = load_embeddings(embedding_dir)
#
#     # === 🛠️ Retry mechanism here ===
#     attempts = 0
#     while attempts < RESTART_LIMIT:
#         try:
#             logger.info(f"🚀 Attempt {attempts+1}/{RESTART_LIMIT} for camera: {camera.name}")
#             process_camera_stream(camera, schedules, face_analyzer, embeddings_map)
#             break
#         except Exception as e:
#             logger.exception(f"🔥 [ERROR] Exception while processing camera {camera.name}: {e}")
#             attempts += 1
#             if attempts < RESTART_LIMIT:
#                 logger.info(f"🔁 Retry {attempts}/{RESTART_LIMIT} after {RESTART_DELAY} seconds")
#                 time.sleep(RESTART_DELAY)
#             else:
#                 logger.error(f"🛑 Max retries reached for camera {camera.name}. Giving up.")
#
# def recognize_and_log():
#     print("🔧 Loading active cameras and recognition schedules...")
#
#     embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")
#
#     now = datetime.now()
#     today_weekday = now.strftime('%a')[:3]  # e.g., 'Mon', 'Tue'
#
#     active_cameras = []
#     camera_schedules_map = {}
#
#     for camera in Camera.objects.filter(is_active=True):
#         schedules = RecognitionSchedule.objects.filter(is_active=True, cameras=camera)
#
#         # Filter for valid current schedules
#         valid_schedules = [
#             s for s in schedules
#             if today_weekday in s.weekdays and s.start_time <= now.time() <= s.end_time
#         ]
#
#         if valid_schedules:
#             active_cameras.append(camera)
#             camera_schedules_map[camera.id] = valid_schedules
#
#     processes = []
#     for camera in active_cameras:
#         print(f"🚀 Spawning process for camera: {camera.name}")
#         p = multiprocessing.Process(target=run_camera, args=(camera, camera_schedules_map[camera.id], embedding_dir))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
#
#     print("✅ All camera processes have completed.")
#
# if __name__ == "__main__":
#     print("🚀 Starting recognition and log attendance (parallel)...")
#     recognize_and_log()





















# # extras/recognize_and_log_attendance_parallel.py (Here InsightFace handles defining the det_set Automatically)
# import os
# import sys
# import django
# import logging
# from datetime import datetime
# import multiprocessing
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
# django.setup()
#
# from attendance.models import Camera, RecognitionSchedule
# from extras.utils import process_camera_stream, is_within_recognition_schedule, load_embeddings, RESTART_LIMIT, RESTART_DELAY
#
# def run_camera(camera, schedules, embedding_dir):
#     from insightface.app import FaceAnalysis  # avoid GPU context collision
#     from extras.log_utils import get_camera_logger  # <-- Move import here too
#     import time
#
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🎥 [START] Processing stream for: {camera.name}")
#
#     # Initialize FaceAnalyzer without explicitly setting the det_size for automatic resizing
#     face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#
#     # f you want slightly better accuracy for very small faces, you could increase it to 768×768 or 800×800, but this will slow down processing.
#     # 640×640 is a good balance (speed vs accuracy).
#     # face_analyzer.prepare(ctx_id=0, det_size=(800, 800))
#     # face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))
#     # face_analyzer.prepare(ctx_id=0, det_size=(1600, 1600))
#     # face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))
#     # face_analyzer.prepare(ctx_id=0, det_size=(2144, 2144))
#
#     # Allow the model to handle resizing automatically instead of specifying det_size
#     face_analyzer.prepare(ctx_id=0)
#
#     embeddings_map = load_embeddings(embedding_dir)
#
#     # === 🛠️ Retry mechanism here ===
#     attempts = 0
#     while attempts < RESTART_LIMIT:
#         try:
#             process_camera_stream(camera, schedules, face_analyzer, embeddings_map)
#             break
#         except Exception as e:
#             logger.exception(f"🔥 [ERROR] Exception while processing camera {camera.name}: {e}")
#             attempts += 1
#             if attempts < RESTART_LIMIT:
#                 logger.info(f"🔁 Retry {attempts}/{RESTART_LIMIT} after {RESTART_DELAY} seconds")
#                 time.sleep(RESTART_DELAY)
#             else:
#                 logger.error(f"🛑 Max retries reached for camera {camera.name}. Giving up.")
#
# def recognize_and_log():
#     print("🔧 Loading active cameras and recognition schedules...")
#
#     embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")
#
#     now = datetime.now()
#     today_weekday = now.strftime('%a')[:3]  # e.g., 'Mon', 'Tue'
#
#     active_cameras = []
#     camera_schedules_map = {}
#
#     for camera in Camera.objects.filter(is_active=True):
#         schedules = RecognitionSchedule.objects.filter(is_active=True, cameras=camera)
#
#         # Filter for valid current schedules
#         valid_schedules = [
#             s for s in schedules
#             if today_weekday in s.weekdays and s.start_time <= now.time() <= s.end_time
#         ]
#
#         if valid_schedules:
#             active_cameras.append(camera)
#             camera_schedules_map[camera.id] = valid_schedules
#
#     processes = []
#     for camera in active_cameras:
#         print(f"🚀 Spawning process for camera: {camera.name}")
#         p = multiprocessing.Process(target=run_camera, args=(camera, camera_schedules_map[camera.id], embedding_dir))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
#
#     print("✅ All camera processes have completed.")
#
# if __name__ == "__main__":
#     print("🚀 Starting recognition and log attendance (parallel)...")
#     recognize_and_log()















# # extras/recognize_and_log_attendance_parallel.py (Here InsightFace handles defining the det_set Automatically)
# import os
# import sys
# import django
# import logging
# from datetime import datetime
# import multiprocessing
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
# django.setup()
#
# from attendance.models import Camera, RecognitionSchedule
# from extras.utils import process_camera_stream, is_within_recognition_schedule, load_embeddings
#
#
# def run_camera(camera, schedules, embedding_dir):
#     from insightface.app import FaceAnalysis  # avoid GPU context collision
#     from extras.log_utils import get_camera_logger  # <-- Move import here too
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🎥 [START] Processing stream for: {camera.name}")
#
#     # Initialize FaceAnalyzer without explicitly setting the det_size for automatic resizing
#     face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#
#     # f you want slightly better accuracy for very small faces, you could increase it to 768×768 or 800×800, but this will slow down processing.
#     # 640×640 is a good balance (speed vs accuracy).
#     # face_analyzer.prepare(ctx_id=0, det_size=(800, 800))
#     # face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))
#     # face_analyzer.prepare(ctx_id=0, det_size=(1600, 1600))
#     # face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))
#     # face_analyzer.prepare(ctx_id=0, det_size=(2144, 2144))
#
#     # Allow the model to handle resizing automatically instead of specifying det_size
#     face_analyzer.prepare(ctx_id=0)
#
#     embeddings_map = load_embeddings(embedding_dir)
#
#     try:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map)
#     except Exception as e:
#         logger.exception(f"🔥 [ERROR] Exception while processing camera {camera.name}: {e}")
#
#
# def recognize_and_log():
#     print("🔧 Loading active cameras and recognition schedules...")
#
#     embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")
#
#     now = datetime.now()
#     today_weekday = now.strftime('%a')[:3]  # e.g., 'Mon', 'Tue'
#
#     active_cameras = []
#     camera_schedules_map = {}
#
#     for camera in Camera.objects.filter(is_active=True):
#         schedules = RecognitionSchedule.objects.filter(is_active=True, cameras=camera)
#
#         # Filter for valid current schedules
#         valid_schedules = [
#             s for s in schedules
#             if today_weekday in s.weekdays and s.start_time <= now.time() <= s.end_time
#         ]
#
#         if valid_schedules:
#             active_cameras.append(camera)
#             camera_schedules_map[camera.id] = valid_schedules
#
#     processes = []
#     for camera in active_cameras:
#         print(f"🚀 Spawning process for camera: {camera.name}")
#         p = multiprocessing.Process(target=run_camera, args=(camera, camera_schedules_map[camera.id], embedding_dir))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
#
#     print("✅ All camera processes have completed.")
#
#
# if __name__ == "__main__":
#     print("🚀 Starting recognition and log attendance (parallel)...")
#     recognize_and_log()






# # extras/recognize_and_log_attendance_parallel.py (Here we define det_set manually)
# import os
# import sys
# import django
# import logging
# from datetime import datetime
# import multiprocessing
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
# django.setup()
#
# from attendance.models import Camera, RecognitionSchedule
# from extras.utils import process_camera_stream, is_within_recognition_schedule, load_embeddings
#
#
# def run_camera(camera, schedules, embedding_dir):
#     from insightface.app import FaceAnalysis  # avoid GPU context collision
#     from extras.log_utils import get_camera_logger  # <-- Move import here too
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🎥 [START] Processing stream for: {camera.name}")
#
#     face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     # 640×640 is a good balance (speed vs accuracy).
#     # f you want slightly better accuracy for very small faces, you could increase it to 768×768 or 800×800, but this will slow down processing.
#     # face_analyzer.prepare(ctx_id=0, det_size=(800, 800))
#     # face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))
#     # face_analyzer.prepare(ctx_id=0, det_size=(1600, 1600))
#     # face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))
#     face_analyzer.prepare(ctx_id=0, det_size=(2144, 2144))
#
#     embeddings_map = load_embeddings(embedding_dir)
#
#     try:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map)
#     except Exception as e:
#         logger.exception(f"🔥 [ERROR] Exception while processing camera {camera.name}: {e}")
#
#
# def recognize_and_log():
#     print("🔧 Loading active cameras and recognition schedules...")
#
#     embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")
#
#     now = datetime.now()
#     today_weekday = now.strftime('%a')[:3]  # e.g., 'Mon', 'Tue'
#
#     active_cameras = []
#     camera_schedules_map = {}
#
#     for camera in Camera.objects.filter(is_active=True):
#         schedules = RecognitionSchedule.objects.filter(is_active=True, cameras=camera)
#
#         # Filter for valid current schedules
#         valid_schedules = [
#             s for s in schedules
#             if today_weekday in s.weekdays and s.start_time <= now.time() <= s.end_time
#         ]
#
#         if valid_schedules:
#             active_cameras.append(camera)
#             camera_schedules_map[camera.id] = valid_schedules
#
#     processes = []
#     for camera in active_cameras:
#         print(f"🚀 Spawning process for camera: {camera.name}")
#         p = multiprocessing.Process(target=run_camera, args=(camera, camera_schedules_map[camera.id], embedding_dir))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
#
#     print("✅ All camera processes have completed.")
#
#
# if __name__ == "__main__":
#     print("🚀 Starting recognition and log attendance (parallel)...")
#     recognize_and_log()









# # extras/recognize_and_log_attendance_parallel.py
# import os
# import sys
# import django
# import logging
# from datetime import datetime
# import multiprocessing
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
# django.setup()
#
# from attendance.models import Camera, RecognitionSchedule
# from extras.utils import process_camera_stream, is_within_recognition_schedule, load_embeddings
#
#
# def run_camera(camera, schedules, embedding_dir):
#     from insightface.app import FaceAnalysis  # avoid GPU context collision
#     from extras.log_utils import get_camera_logger  # <-- Move import here too
#     logger = get_camera_logger(camera.name)
#     logger.info(f"🎥 [START] Processing stream for: {camera.name}")
#
#     face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     # 640×640 is a good balance (speed vs accuracy).
#     # f you want slightly better accuracy for very small faces, you could increase it to 768×768 or 800×800, but this will slow down processing.
#     face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
#     # face_analyzer.prepare(ctx_id=0, det_size=(800, 800))
#
#     embeddings_map = load_embeddings(embedding_dir)
#
#     try:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map)
#     except Exception as e:
#         logger.exception(f"🔥 [ERROR] Exception while processing camera {camera.name}: {e}")
#
# # def recognize_and_log():
# #     logger.info("🔧 Loading active cameras and recognition schedules...")
# #
# #     embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")
# #     active_cameras = Camera.objects.filter(is_active=True)
# #     schedules = RecognitionSchedule.objects.filter(is_active=True)
# #
# #     camera_schedules_map = {
# #         cam.id: [s for s in schedules if cam in s.cameras.all()]
# #         for cam in active_cameras
# #     }
# #
# #     processes = []
# #     for camera in active_cameras:
# #         logger.info(f"🚀 Spawning process for camera: {camera.name}")
# #         p = multiprocessing.Process(target=run_camera, args=(camera, camera_schedules_map[camera.id], embedding_dir))
# #         p.start()
# #         processes.append(p)
# #
# #     for p in processes:
# #         p.join()
# #
# #     logger.info("✅ All camera processes have completed.")
#
# def recognize_and_log():
#     print("🔧 Loading active cameras and recognition schedules...")
#
#     embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")
#
#     # active_cameras = Camera.objects.filter(is_active=True)
#     # schedules = RecognitionSchedule.objects.filter(is_active=True)
#     # camera_schedules_map = {
#     #     cam.id: [s for s in schedules if cam in s.cameras.all()]
#     #     for cam in active_cameras
#     # }
#
#     now = datetime.now()
#     today_weekday = now.strftime('%a')[:3]  # e.g., 'Mon', 'Tue'
#
#     active_cameras = []
#     camera_schedules_map = {}
#
#     for camera in Camera.objects.filter(is_active=True):
#         schedules = RecognitionSchedule.objects.filter(is_active=True, cameras=camera)
#
#         # Filter for valid current schedules
#         valid_schedules = [
#             s for s in schedules
#             if today_weekday in s.weekdays and s.start_time <= now.time() <= s.end_time
#         ]
#
#         if valid_schedules:
#             active_cameras.append(camera)
#             camera_schedules_map[camera.id] = valid_schedules
#
#     processes = []
#     for camera in active_cameras:
#         print(f"🚀 Spawning process for camera: {camera.name}")
#         p = multiprocessing.Process(target=run_camera, args=(camera, camera_schedules_map[camera.id], embedding_dir))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
#
#     print("✅ All camera processes have completed.")
#
#
# # if __name__ == "__main__":
# #     logger.info("🚀 Starting recognition and log attendance (parallel)...")
# #     recognize_and_log()
# if __name__ == "__main__":
#     print("🚀 Starting recognition and log attendance (parallel)...")
#     recognize_and_log()
#
#


#
# # extras/recognize_and_log_attendance_parallel.py
# import os
# import sys
# import django
# import logging
# from datetime import datetime
# import multiprocessing
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
# django.setup()
#
# from attendance.models import Camera, RecognitionSchedule
# from extras.utils import process_camera_stream, is_within_recognition_schedule, load_embeddings
#
# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)
#
# def run_camera(camera, schedules, embedding_dir):
#     from insightface.app import FaceAnalysis  # avoid GPU context collision
#     logger.info(f"🎥 [START] Processing stream for: {camera.name}")
#
#     face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
#
#     embeddings_map = load_embeddings(embedding_dir)
#
#     try:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map)
#     except Exception as e:
#         logger.exception(f"🔥 [ERROR] Exception while processing camera {camera.name}: {e}")
#
# def recognize_and_log():
#     logger.info("🔧 Loading active cameras and recognition schedules...")
#
#     embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")
#     active_cameras = Camera.objects.filter(is_active=True)
#     schedules = RecognitionSchedule.objects.filter(is_active=True)
#
#     camera_schedules_map = {
#         cam.id: [s for s in schedules if cam in s.cameras.all()]
#         for cam in active_cameras
#     }
#
#     processes = []
#     for camera in active_cameras:
#         logger.info(f"🚀 Spawning process for camera: {camera.name}")
#         p = multiprocessing.Process(target=run_camera, args=(camera, camera_schedules_map[camera.id], embedding_dir))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
#
#     logger.info("✅ All camera processes have completed.")
#
# if __name__ == "__main__":
#     logger.info("🚀 Starting recognition and log attendance (parallel)...")
#     recognize_and_log()












# # recognize_and_log_attendance_parallel.py
# # --- Django setup block (MUST BE FIRST) ---
# import os
# import sys
# import django
# from datetime import datetime
# import logging
# import multiprocessing
# import pytz
#
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(BASE_DIR)
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
# django.setup()
# # ------------------------------------------
#
# # Now it's safe to import anything Django-related
# from attendance.models import Camera, RecognitionSchedule
# from extras.utils import process_camera_stream, is_within_recognition_schedule, load_embeddings
#
# # Setup timezone
# IRAQ_TZ = pytz.timezone("Asia/Baghdad")
#
# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='[%(asctime)s] %(levelname)s - %(message)s',
#     datefmt='%Y-%m-%d %H:%M:%S'
# )
# logger = logging.getLogger(__name__)
#
# def run_camera(camera, schedules, embedding_dir):
#     from insightface.app import FaceAnalysis  # avoid GPU context collision
#     logger.info(f"🎥 [START] Processing stream for: {camera.name}")
#
#     face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
#
#     embeddings_map = load_embeddings(embedding_dir)
#
#     try:
#         process_camera_stream(camera, schedules, face_analyzer, embeddings_map)
#     except Exception as e:
#         logger.exception(f"🔥 [ERROR] Exception while processing camera {camera.name}: {e}")
#
# def recognize_and_log():
#     logger.info("🔧 Loading active cameras and recognition schedules...")
#
#     embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")
#     active_cameras = Camera.objects.filter(is_active=True)
#     schedules = RecognitionSchedule.objects.filter(is_active=True)
#
#     camera_schedules_map = {
#         cam.id: [s for s in schedules if cam in s.cameras.all()]
#         for cam in active_cameras
#     }
#
#     processes = []
#     for camera in active_cameras:
#         logger.info(f"🚀 Spawning process for camera: {camera.name}")
#         p = multiprocessing.Process(target=run_camera, args=(camera, camera_schedules_map[camera.id], embedding_dir))
#         p.start()
#         processes.append(p)
#
#     for p in processes:
#         p.join()
#
#     logger.info("✅ All camera processes have completed.")
#
# if __name__ == "__main__":
#     logger.info("🚀 Starting recognition and log attendance (parallel)...")
#     recognize_and_log()
