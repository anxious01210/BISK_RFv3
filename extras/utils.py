# extras/utils.py
import cv2
import numpy as np
from datetime import datetime
from django.utils.timezone import now
from attendance.models import AttendanceLog, Period, RecognitionSchedule, Student
from django.utils import timezone

def is_within_recognition_schedule(current_time, schedule):
    weekday = current_time.strftime('%a')  # 'Mon', 'Tue', etc.
    return (
        schedule.is_active and
        weekday in schedule.weekdays and
        schedule.start_time <= current_time.time() <= schedule.end_time
    )

def process_camera_stream(camera, schedules, face_analyzer, embeddings_map):
    # You’ll eventually move the streaming and recognition logic here
    print(f"🟢 Processing camera: {camera.name}")
    # TODO: Implement the real stream recognition here
    pass
