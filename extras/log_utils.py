# extras/log_utils.py
import os
import logging
from logging.handlers import TimedRotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'attendance')
os.makedirs(LOG_DIR, exist_ok=True)

def get_camera_logger(camera_name):
    logger = logging.getLogger(f'attendance.{camera_name}')
    if logger.hasHandlers():
        return logger  # Prevent multiple handlers

    logger.setLevel(logging.INFO)
    log_path = os.path.join(LOG_DIR, f'{camera_name}.log')

    handler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=14)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger
