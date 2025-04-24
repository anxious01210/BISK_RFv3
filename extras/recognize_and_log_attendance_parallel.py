import os
import django
import sys

# Setup Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'BISK_RFv3.settings')
django.setup()

import multiprocessing
from datetime import datetime
import pytz
from attendance.models import Camera, RecognitionSchedule
from extras.utils import process_camera_stream, is_within_recognition_schedule
from django.utils.timezone import now

# Timezone for Iraq
IRAQ_TZ = pytz.timezone("Asia/Baghdad")

from insightface.app import FaceAnalysis
import numpy as np
import pickle

# Initialize the InsightFace analyzer with GPU support
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

# Load precomputed embeddings from disk (adjust path if needed)
embedding_dir = os.path.join(os.path.dirname(__file__), '..', 'media', 'embeddings')
embeddings_map = {}

for fname in os.listdir(embedding_dir):
    if fname.endswith('.pkl'):
        student_code = fname.replace('.pkl', '')
        with open(os.path.join(embedding_dir, fname), 'rb') as f:
            embeddings_map[student_code] = pickle.load(f)


def recognize_and_log():
    # Get all active cameras
    cameras = Camera.objects.filter(is_active=True)

    # Get all active recognition schedules
    active_schedules = RecognitionSchedule.objects.filter(is_active=True)

    # 🔧 Map schedules to each camera
    camera_schedules_map = {
        cam.id: [s for s in active_schedules if cam in s.cameras.all()]
        for cam in cameras
    }

    def should_process(camera):
        current_time = now().astimezone(IRAQ_TZ)
        for schedule in active_schedules:
            if camera in schedule.cameras.all() and is_within_recognition_schedule(current_time, schedule):
                return True
        return False

    processes = []

    for camera in cameras:
        if should_process(camera):
            p = multiprocessing.Process(
                target=process_camera_stream,
                args=(camera, camera_schedules_map[camera.id], face_analyzer, embeddings_map)
            )
            p.start()
            processes.append(p)

    for p in processes:
        p.join()

if __name__ == "__main__":
    recognize_and_log()
