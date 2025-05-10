# extras/recognize_and_log_attendance_ffmpeg_parallel.py
import os
import sys
import django
import multiprocessing
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
django.setup()

from attendance.models import Camera, RecognitionSchedule
from extras.utils_ffmpeg import process_camera_stream_ffmpeg, RESTART_LIMIT, RESTART_DELAY, load_embeddings


def run_camera_ffmpeg(camera, schedules, embedding_dir):
    from insightface.app import FaceAnalysis
    from extras.log_utils import get_camera_logger
    import time

    logger = get_camera_logger(camera.name)
    logger.info(f"🎥 [START] FFmpeg stream for camera: {camera.name}")

    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # If you want slightly better accuracy for very small faces, you could increase it to 768×768 or 800×800, but this will slow down processing.
    # 640×640 is a good balance (speed vs accuracy).
    # face_analyzer.prepare(ctx_id=0, det_size=(800, 800))
    # face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))
    # face_analyzer.prepare(ctx_id=0, det_size=(1600, 1600))
    # face_analyzer.prepare(ctx_id=0, det_size=(2048, 2048))
    # face_analyzer.prepare(ctx_id=0, det_size=(2144, 2144))
    # face_analyzer.prepare(ctx_id=0, det_size=(3840, 3840))

    # ✅ Allow the model to handle resizing automatically instead of specifying det_size
    face_analyzer.prepare(ctx_id=0)

    embeddings_map = load_embeddings(embedding_dir)

    attempts = 0
    while attempts < RESTART_LIMIT:
        try:
            logger.info(f"🚀 FFmpeg attempt {attempts+1}/{RESTART_LIMIT} for {camera.name}")
            process_camera_stream_ffmpeg(camera, schedules, face_analyzer, embeddings_map)
            break
        except Exception as e:
            logger.exception(f"🔥 [FFmpeg ERROR] Camera {camera.name}: {e}")
            attempts += 1
            if attempts < RESTART_LIMIT:
                logger.info(f"🔁 FFmpeg retry {attempts}/{RESTART_LIMIT} after {RESTART_DELAY}s")
                time.sleep(RESTART_DELAY)
            else:
                logger.error(f"🛑 Max FFmpeg retries reached for {camera.name}. Giving up.")


def recognize_and_log_ffmpeg():
    print("🔧 Loading cameras and recognition schedules for FFmpeg mode...")
    embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")

    now = datetime.now()
    today_weekday = now.strftime('%a')[:3]

    active_cameras = []
    camera_schedules_map = {}

    for camera in Camera.objects.filter(is_active=True):
        # schedules = RecognitionSchedule.objects.filter(is_active=True, cameras=camera).select_related('period')
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
        print(f"🚀 FFmpeg spawning process for: {camera.name}")
        p = multiprocessing.Process(target=run_camera_ffmpeg, args=(camera, camera_schedules_map[camera.id], embedding_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("✅ All FFmpeg-based camera processes have completed.")


if __name__ == "__main__":
    print("🚀 Starting FFmpeg-based parallel face recognition and logging...")
    recognize_and_log_ffmpeg()