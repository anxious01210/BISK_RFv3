# dashboard/scripts.py
import subprocess
import os
import signal
import time

SCRIPT_PROCESS = None
SCRIPT_START_TIME = None
SCRIPT_TYPE = None
DET_SET = None

def run_script_by_type(script_type, det_set):
    global SCRIPT_PROCESS, SCRIPT_START_TIME, SCRIPT_TYPE, DET_SET
    stop_running_scripts()
    script_map = {
        'opencv': 'extras/recognize_and_log_attendance_parallel.py',
        'ffmpeg': 'extras/recognize_and_log_attendance_ffmpeg_parallel.py',
    }
    script_path = script_map.get(script_type)
    if script_path:
        args = ['python3', script_path]
        if det_set != 'auto':
            args += ['--det_set', det_set]
        # SCRIPT_PROCESS = subprocess.Popen(args)
        SCRIPT_PROCESS = subprocess.Popen(args, preexec_fn=os.setsid)
        SCRIPT_START_TIME = time.time()
        SCRIPT_TYPE = script_type
        DET_SET = det_set

def stop_running_scripts():
    global SCRIPT_PROCESS, SCRIPT_START_TIME
    if SCRIPT_PROCESS and SCRIPT_PROCESS.poll() is None:
        # os.kill(SCRIPT_PROCESS.pid, signal.SIGTERM)
        os.killpg(os.getpgid(SCRIPT_PROCESS.pid), signal.SIGTERM)
    SCRIPT_PROCESS = None
    SCRIPT_START_TIME = None

def get_running_script_info():
    if SCRIPT_PROCESS and SCRIPT_PROCESS.poll() is None:
        return {
            'running': True,
            'uptime': time.time() - SCRIPT_START_TIME,
            'type': SCRIPT_TYPE,
            'det_set': DET_SET,
        }
    return {
        'running': False
    }



# # dashboard/scripts.py
# import subprocess
#
# def run_script_by_type(script_type, det_set):
#     script_map = {
#         'opencv': 'extras/recognize_and_log_attendance_parallel.py',
#         'ffmpeg': 'extras/recognize_and_log_attendance_ffmpeg_parallel.py',
#     }
#     script_path = script_map.get(script_type)
#     if script_path:
#         args = ['python3', script_path]
#         if det_set != 'auto':
#             args += ['--det_set', det_set]
#         subprocess.Popen(args)
#



# # dashboard/scripts.py
# import subprocess
#
# def run_script_by_type(script_type, det_set):
#     script_map = {
#         1: 'extras/recognize_and_log_attendance_parallel.py',
#         2: 'extras/recognize_and_log_attendance_ffmpeg_parallel.py',
#     }
#     script_path = script_map.get(script_type)
#     if script_path:
#         subprocess.Popen(['python3', script_path, '--det_set', det_set])