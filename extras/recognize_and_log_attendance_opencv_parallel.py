# extras/recognize_and_log_attendance_opencv_parallel.py
import os
import sys
import django
import multiprocessing
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
django.setup()

from django.db import connections
connections.close_all()

from attendance.models import Camera, RecognitionSchedule
from extras.utils_opencv import process_camera_stream_opencv, RESTART_LIMIT, RESTART_DELAY, load_embeddings

def run_camera_opencv(camera, schedules, embedding_dir):
    from insightface.app import FaceAnalysis
    from extras.log_utils import get_camera_logger
    import time

    connections.close_all()

    logger = get_camera_logger(camera.name)
    logger.info(f"🎥 [START] OpenCV stream for camera: {camera.name}")

    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    det_set_env = os.environ.get("DET_SET", "auto")
    if det_set_env.lower() == "auto":
        face_analyzer.prepare(ctx_id=0)
    else:
        try:
            w, h = map(int, det_set_env.split(","))
            face_analyzer.prepare(ctx_id=0, det_size=(w, h))
        except Exception as e:
            logger.warning(f"⚠️ Invalid DET_SET '{det_set_env}', falling back to auto. Error: {e}")
            face_analyzer.prepare(ctx_id=0)

    embeddings_map = load_embeddings(embedding_dir)

    attempts = 0
    while attempts < RESTART_LIMIT:
        try:
            logger.info(f"🚀 OpenCV attempt {attempts+1}/{RESTART_LIMIT} for {camera.name}")
            process_camera_stream_opencv(camera, schedules, face_analyzer, embeddings_map)
            break
        except Exception as e:
            logger.exception(f"🔥 [OpenCV ERROR] Camera {camera.name}: {e}")
            attempts += 1
            if attempts < RESTART_LIMIT:
                logger.info(f"🔁 OpenCV retry {attempts}/{RESTART_LIMIT} after {RESTART_DELAY}s")
                time.sleep(RESTART_DELAY)
            else:
                logger.error(f"🛑 Max OpenCV retries reached for {camera.name}. Giving up.")

def recognize_and_log_opencv():
    print("🔧 Loading cameras and recognition schedules for OpenCV mode...")
    embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")

    now = datetime.now()
    today_weekday = now.strftime('%a')[:3]

    active_cameras = []
    camera_schedules_map = {}

    for camera in Camera.objects.filter(is_active=True):
        schedules = RecognitionSchedule.objects.filter(is_active=True, cameras=camera)
        valid_schedules = [
            s for s in schedules
            if today_weekday in s.weekdays
        ]
        if valid_schedules:
            active_cameras.append(camera)
            camera_schedules_map[camera.id] = valid_schedules

    processes = []
    for camera in active_cameras:
        print(f"🚀 OpenCV spawning process for: {camera.name}")
        p = multiprocessing.Process(target=run_camera_opencv, args=(camera, camera_schedules_map[camera.id], embedding_dir), daemon=False)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        print(f"📄 Process {p.name} (PID {p.pid}) exited with code {p.exitcode}")

    print("✅ All OpenCV-based camera processes have completed.")

if __name__ == "__main__":
    from django.db import connections
    connections.close_all()
    print("🚀 Starting OpenCV-based parallel face recognition and logging...")
    recognize_and_log_opencv()