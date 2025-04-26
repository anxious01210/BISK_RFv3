# recognize_and_log_attendance_parallel.py
# --- Django setup block (MUST BE FIRST) ---
import os
import sys
import django
from datetime import datetime
import logging
import multiprocessing
import pytz

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
django.setup()
# ------------------------------------------

# Now it's safe to import anything Django-related
from attendance.models import Camera, RecognitionSchedule
from extras.utils import process_camera_stream, is_within_recognition_schedule, load_embeddings

# Setup timezone
IRAQ_TZ = pytz.timezone("Asia/Baghdad")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def run_camera(camera, schedules, embedding_dir):
    from insightface.app import FaceAnalysis  # avoid GPU context collision
    logger.info(f"🎥 [START] Processing stream for: {camera.name}")

    face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

    embeddings_map = load_embeddings(embedding_dir)

    try:
        process_camera_stream(camera, schedules, face_analyzer, embeddings_map)
    except Exception as e:
        logger.exception(f"🔥 [ERROR] Exception while processing camera {camera.name}: {e}")

def recognize_and_log():
    logger.info("🔧 Loading active cameras and recognition schedules...")

    embedding_dir = os.path.join(BASE_DIR, "media", "embeddings")
    active_cameras = Camera.objects.filter(is_active=True)
    schedules = RecognitionSchedule.objects.filter(is_active=True)

    camera_schedules_map = {
        cam.id: [s for s in schedules if cam in s.cameras.all()]
        for cam in active_cameras
    }

    processes = []
    for camera in active_cameras:
        logger.info(f"🚀 Spawning process for camera: {camera.name}")
        p = multiprocessing.Process(target=run_camera, args=(camera, camera_schedules_map[camera.id], embedding_dir))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    logger.info("✅ All camera processes have completed.")

if __name__ == "__main__":
    logger.info("🚀 Starting recognition and log attendance (parallel)...")
    recognize_and_log()
