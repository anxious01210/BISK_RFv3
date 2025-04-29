# extras/log_utils.py
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from django.conf import settings
from datetime import datetime

def get_camera_logger(camera_name):
    logger = logging.getLogger(f'attendance.{camera_name}')

    if logger.hasHandlers():
        return logger

    logger.setLevel(logging.INFO)

    log_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'attendance')
    os.makedirs(log_dir, exist_ok=True)

    today = datetime.now().strftime("%Y-%m-%d")
    log_path = os.path.join(log_dir, f"{camera_name}_{today}.log")

    handler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=31, encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    return logger





# # extras/log_utils.py
# import os
# import logging
# from logging.handlers import TimedRotatingFileHandler
# from django.conf import settings
# from datetime import date, datetime
#
# def get_camera_logger(camera_name):
#     logger = logging.getLogger(f'attendance.{camera_name}')
#
#     if logger.hasHandlers():
#         return logger
#
#     logger.setLevel(logging.INFO)
#
#     log_dir = os.path.join(settings.MEDIA_ROOT, 'logs', 'attendance')
#     os.makedirs(log_dir, exist_ok=True)
#
#     # today_str = date.today().strftime('%Y-%m-%d')
#     # log_path = os.path.join(log_dir, f'{camera_name}_{today_str}.log')
#
#     today = datetime.now().strftime("%Y-%m-%d")
#     log_path = f"media/logs/attendance/{camera.name}_{today}.log"
#
#     handler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=31, encoding='utf-8')
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#
#     logger.addHandler(handler)
#     return logger











# # extras/log_utils.py
# import os
# import logging
# from logging.handlers import TimedRotatingFileHandler
# from django.conf import settings
# from datetime import date
#
# LOG_DIR = os.path.join(settings.MEDIA_ROOT, 'logs', 'attendance')
# os.makedirs(LOG_DIR, exist_ok=True)
#
# def get_camera_logger(camera_name):
#     logger = logging.getLogger(f'attendance.{camera_name}')
#     if logger.hasHandlers():
#         return logger
#
#     logger.setLevel(logging.INFO)
#     today_str = date.today().strftime('%Y-%m-%d')
#     log_path = os.path.join(LOG_DIR, f'{camera_name}_{today_str}.log')
#
#     handler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=31, encoding='utf-8')
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#
#     logger.addHandler(handler)
#     return logger











# # extras/log_utils.py
# import os
# import logging
# from datetime import date
# from logging.handlers import TimedRotatingFileHandler
# from django.conf import settings
#
# # Use Django MEDIA_ROOT
# LOG_DIR = os.path.join(settings.MEDIA_ROOT, 'logs', 'attendance')
# os.makedirs(LOG_DIR, exist_ok=True)
#
# def get_camera_logger(camera_name):
#     logger = logging.getLogger(f'attendance.{camera_name}')
#     if logger.hasHandlers():
#         return logger  # Prevent multiple handlers
#
#     logger.setLevel(logging.INFO)
#     log_path = os.path.join(LOG_DIR, f'{camera_name}_{date.today()}.log')
#
#     handler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=14)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#
#     logger.addHandler(handler)
#     return logger




# # extras/log_utils.py
# import os
# import logging
# from logging.handlers import TimedRotatingFileHandler
# from django.conf import settings
#
# LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'attendance')
# os.makedirs(LOG_DIR, exist_ok=True)
#
# def get_camera_logger(camera_name):
#     today_str = datetime.now().strftime("%Y-%m-%d")
#     log_filename = f"{camera_name}_{today_str}.log"
#     log_path = os.path.join(LOG_DIR, log_filename)
#
#     logger = logging.getLogger(f'attendance.{camera_name}')
#     logger.propagate = False  # Prevent duplicate logging
#
#     if not logger.handlers:
#         handler = logging.FileHandler(log_path, mode='a')
#         formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#         handler.setFormatter(formatter)
#         logger.setLevel(logging.INFO)
#         logger.addHandler(handler)
#
#     return logger





# # extras/log_utils.py
# import os
# import logging
# from logging.handlers import TimedRotatingFileHandler
#
# LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'attendance')
# os.makedirs(LOG_DIR, exist_ok=True)
#
# def get_camera_logger(camera_name):
#     logger = logging.getLogger(f'attendance.{camera_name}')
#     if logger.hasHandlers():
#         return logger  # Prevent multiple handlers
#
#     logger.setLevel(logging.INFO)
#     log_path = os.path.join(LOG_DIR, f'{camera_name}.log')
#
#     handler = TimedRotatingFileHandler(log_path, when='midnight', backupCount=14)
#     formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
#     handler.setFormatter(formatter)
#
#     logger.addHandler(handler)
#     return logger
