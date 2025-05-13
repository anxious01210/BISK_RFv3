# dashboard/views.py
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseRedirect
from .scripts import run_script_by_type, stop_running_scripts, get_running_script_info
from .utils import get_system_stats
import time
import platform
from datetime import timedelta
from django.urls import reverse
from django.contrib.auth.decorators import login_required
import subprocess
import signal
import os
from attendance.models import Camera

DJANGO_START_TIME = time.time()

def stop_glances():
    try:
        output = subprocess.check_output(['lsof', '-t', '-i:61208']).decode().strip()
        for pid in output.splitlines():
            os.kill(int(pid), signal.SIGTERM)
            time.sleep(0.5)
    except subprocess.CalledProcessError:
        pass
    time.sleep(1.5)

def start_glances():
    stop_glances()  # always stop first
    with open('/tmp/glances.log', 'w') as f:
        subprocess.Popen(
            ['/home/rio/PycharmProjects/BISK_RFv3/.venv/bin/glances', '-w', '-B', '127.0.0.1', '-p', '61208'],
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )
    time.sleep(1.5)

def is_glances_running():
    try:
        output = subprocess.check_output(['lsof', '-nP', '-iTCP:61208', '-sTCP:LISTEN'], stderr=subprocess.DEVNULL)
        return 'glances' in output.decode().lower()
    except subprocess.CalledProcessError:
        return False

@login_required
def dashboard_view(request):
    selected_script_type = 'ffmpeg'
    selected_det_set = 'auto'

    if request.method == 'POST':
        if 'start_glances' in request.POST:
            start_glances()
        elif 'stop_glances' in request.POST:
            stop_glances()
        elif 'restart_glances' in request.POST:
            stop_glances()
            start_glances()

        if 'run_button' in request.POST:
            selected_script_type = request.POST.get('script_type', 'opencv')
            selected_det_set = request.POST.get('det_set', 'auto')
            run_script_by_type(selected_script_type, selected_det_set)
        elif 'stop_button' in request.POST:
            stop_running_scripts()

        return HttpResponseRedirect(reverse('dashboard:dashboard'))

    running_info = get_running_script_info()
    context = {
        'running_info': running_info,
        'django_uptime': str(timedelta(seconds=int(time.time() - DJANGO_START_TIME))),
        'selected_script_type': running_info.get('type', 'ffmpeg'),
        'selected_det_set': running_info.get('det_set', 'auto'),
        'det_set_options': [
            'auto', '320,320', '480,480', '640,640', '800,800',
            '1024,1024', '1280,1280', '1600,1600', '1920,1920', '2048,2048'
        ],
        'glances_running': is_glances_running(),
        'camera_statuses': Camera.objects.filter(is_active=True),
    }
    return render(request, 'dashboard/dashboard.html', context)

def system_stats_view(request):
    stats = get_system_stats()
    stats['django_uptime'] = str(timedelta(seconds=int(time.time() - DJANGO_START_TIME)))

    script_info = get_running_script_info()
    if script_info.get('running'):
        uptime = int(script_info.get('uptime', 0))
        script_info['uptime'] = str(timedelta(seconds=uptime))
    stats['script_info'] = script_info
    stats['glances_running'] = is_glances_running()

    return JsonResponse(stats)











# dashboard/views.py
# from django.shortcuts import render
# from django.http import JsonResponse, HttpResponseRedirect
# from .scripts import run_script_by_type, stop_running_scripts, get_running_script_info
# from .utils import get_system_stats
# import time
# import platform
# from datetime import timedelta
# from django.urls import reverse
# from django.contrib.auth.decorators import login_required
# import subprocess
# import signal
# import os
#
# GLANCES_PID_FILE = '/tmp/glances_web.pid'
#
#
# def start_glances():
#     if not os.path.exists(GLANCES_PID_FILE):
#         proc = subprocess.Popen(
#             ['glances', '-w', '-B', '127.0.0.1', '-p', '61208'],
#             # ['/home/rio/PycharmProjects/BISK_RFv3/.venv/bin/glances', '-w', '-B', '127.0.0.1', '-p', '61208'],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#             preexec_fn=os.setsid
#         )
#         with open(GLANCES_PID_FILE, 'w') as f:
#             f.write(str(proc.pid))
#         time.sleep(1.5)  # wait a bit so it's fully booted
#
#
# def stop_glances():
#     if os.path.exists(GLANCES_PID_FILE):
#         with open(GLANCES_PID_FILE, 'r') as f:
#             pid = int(f.read().strip())
#         try:
#             os.killpg(os.getpgid(pid), signal.SIGTERM)
#         except ProcessLookupError:
#             pass
#         os.remove(GLANCES_PID_FILE)
#     time.sleep(1.5)  # allow system to update state
#
#
# def is_glances_running():
#     try:
#         output = subprocess.check_output(
#             ['lsof', '-nP', '-iTCP:61208', '-sTCP:LISTEN'],
#             stderr=subprocess.DEVNULL
#         ).decode()
#         return 'glances' in output.lower()
#     except subprocess.CalledProcessError:
#         return False
#
#
# DJANGO_START_TIME = time.time()
#
#
# @login_required
# def dashboard_view(request):
#     selected_script_type = 'ffmpeg'
#     selected_det_set = 'auto'
#     if 'start_glances' in request.POST:
#         print("▶ Start Glances triggered")
#         start_glances()
#
#     if request.method == 'POST':
#         if 'start_glances' in request.POST:
#             start_glances()
#         elif 'stop_glances' in request.POST:
#             stop_glances()
#         elif 'restart_glances' in request.POST:
#             stop_glances()
#             start_glances()
#
#         if 'run_button' in request.POST:
#             selected_script_type = request.POST.get('script_type', 'opencv')
#             selected_det_set = request.POST.get('det_set', 'auto')
#             run_script_by_type(selected_script_type, selected_det_set)
#         elif 'stop_button' in request.POST:
#             stop_running_scripts()
#
#         return HttpResponseRedirect(reverse('dashboard:dashboard'))
#
#     running_info = get_running_script_info()
#     context = {
#         'running_info': running_info,
#         'django_uptime': str(timedelta(seconds=int(time.time() - DJANGO_START_TIME))),
#         'selected_script_type': running_info.get('type', 'ffmpeg'),
#         'selected_det_set': running_info.get('det_set', 'auto'),
#         'det_set_options': [
#             'auto', '320,320', '480,480', '640,640', '800,800',
#             '1024,1024', '1280,1280', '1600,1600', '1920,1920', '2048,2048'
#         ],
#         'glances_running': is_glances_running(),
#     }
#     return render(request, 'dashboard/dashboard.html', context)
#
#
# def system_stats_view(request):
#     stats = get_system_stats()
#     stats['django_uptime'] = str(timedelta(seconds=int(time.time() - DJANGO_START_TIME)))
#
#     script_info = get_running_script_info()
#     if script_info.get('running'):
#         uptime = int(script_info.get('uptime', 0))
#         script_info['uptime'] = str(timedelta(seconds=uptime))
#     stats['script_info'] = script_info
#
#     stats['glances_running'] = is_glances_running()
#
#     return JsonResponse(stats)







# dashboard/views.py
# from django.shortcuts import render
# from django.http import JsonResponse, HttpResponseRedirect
# from .scripts import run_script_by_type, stop_running_scripts, get_running_script_info
# from .utils import get_system_stats
# import time
# import platform
# from django.urls import reverse
# from django.contrib.auth.decorators import login_required
# import subprocess
# import signal
# import os
# from datetime import timedelta
#
# GLANCES_PID_FILE = '/tmp/glances_web.pid'
#
#
# def start_glances():
#     if not os.path.exists(GLANCES_PID_FILE):
#         proc = subprocess.Popen(
#             ['glances', '-w', '-B', '127.0.0.1', '-p', '61208'],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#             preexec_fn=os.setsid
#         )
#         with open(GLANCES_PID_FILE, 'w') as f:
#             f.write(str(proc.pid))
#         time.sleep(1.5)  # give it a moment to boot up
#
#
# def stop_glances():
#     if os.path.exists(GLANCES_PID_FILE):
#         with open(GLANCES_PID_FILE, 'r') as f:
#             pid = int(f.read().strip())
#         try:
#             os.killpg(os.getpgid(pid), signal.SIGTERM)
#         except ProcessLookupError:
#             pass
#         os.remove(GLANCES_PID_FILE)
#     time.sleep(1.5)  # ensure time for system to reflect process termination
#
#
# def is_glances_running():
#     if os.path.exists(GLANCES_PID_FILE):
#         try:
#             with open(GLANCES_PID_FILE, 'r') as f:
#                 pid = int(f.read().strip())
#             # Check if process exists
#             os.kill(pid, 0)
#             # Check if it's Glances
#             cmdline = subprocess.check_output(['ps', '-p', str(pid), '-o', 'cmd=']).decode().strip()
#             return 'glances' in cmdline.lower()
#         except Exception:
#             return False
#     return False
#
#
# DJANGO_START_TIME = time.time()
#
#
# @login_required
# def dashboard_view(request):
#     selected_script_type = 'ffmpeg'
#     selected_det_set = 'auto'
#
#     if request.method == 'POST':
#         if 'start_glances' in request.POST:
#             start_glances()
#             time.sleep(1.5)
#         elif 'stop_glances' in request.POST:
#             stop_glances()
#         elif 'restart_glances' in request.POST:
#             stop_glances()
#             start_glances()
#             time.sleep(1.5)
#         if 'run_button' in request.POST:
#             selected_script_type = request.POST.get('script_type', 'opencv')
#             selected_det_set = request.POST.get('det_set', 'auto')
#             run_script_by_type(selected_script_type, selected_det_set)
#         elif 'stop_button' in request.POST:
#             stop_running_scripts()
#         return HttpResponseRedirect(reverse('dashboard:dashboard'))
#
#     running_info = get_running_script_info()
#     context = {
#         'running_info': running_info,
#         'django_uptime': str(timedelta(seconds=int(time.time() - DJANGO_START_TIME))),
#         'selected_script_type': running_info.get('type', 'ffmpeg'),
#         'selected_det_set': running_info.get('det_set', 'auto'),
#         'det_set_options': [
#             'auto', '320,320', '480,480', '640,640', '800,800',
#             '1024,1024', '1280,1280', '1600,1600', '1920,1920', '2048,2048'
#         ],
#         'glances_running': is_glances_running(),
#     }
#     return render(request, 'dashboard/dashboard.html', context)
#
#
# def system_stats_view(request):
#     stats = get_system_stats()
#     uptime_seconds = int(time.time() - DJANGO_START_TIME)
#     stats['django_uptime'] = str(timedelta(seconds=uptime_seconds))
#
#     script_info = get_running_script_info()
#     if script_info.get('running'):
#         uptime = int(script_info.get('uptime', 0))
#         script_info['uptime'] = str(timedelta(seconds=uptime))
#     stats['script_info'] = script_info
#     return JsonResponse(stats)







# from django.shortcuts import render
# from django.http import JsonResponse, HttpResponseRedirect
# from .scripts import run_script_by_type, stop_running_scripts, get_running_script_info
# from .utils import get_system_stats
# import time
# import platform
# from django.urls import reverse
# from django.contrib.auth.decorators import login_required
# import subprocess
# import signal
# import os
#
# GLANCES_PID_FILE = '/tmp/glances_web.pid'
#
#
# def start_glances():
#     if not os.path.exists(GLANCES_PID_FILE):
#         proc = subprocess.Popen(
#             ['glances', '-w', '-B', '127.0.0.1', '-p', '61208'],
#             stdout=subprocess.DEVNULL,
#             stderr=subprocess.DEVNULL,
#             preexec_fn=os.setsid
#         )
#         with open(GLANCES_PID_FILE, 'w') as f:
#             f.write(str(proc.pid))
#         time.sleep(1.5)  # give it a moment to boot up
#
#
# def stop_glances():
#     if os.path.exists(GLANCES_PID_FILE):
#         with open(GLANCES_PID_FILE, 'r') as f:
#             pid = int(f.read().strip())
#         try:
#             os.killpg(os.getpgid(pid), signal.SIGTERM)
#         except ProcessLookupError:
#             pass
#         os.remove(GLANCES_PID_FILE)
#     time.sleep(1.5)  # ensure time for system to reflect process termination
#
#
# def is_glances_running():
#     if os.path.exists(GLANCES_PID_FILE):
#         try:
#             with open(GLANCES_PID_FILE, 'r') as f:
#                 pid = int(f.read().strip())
#             # Check if process exists
#             os.kill(pid, 0)
#             # Check if it's Glances
#             cmdline = subprocess.check_output(['ps', '-p', str(pid), '-o', 'cmd=']).decode().strip()
#             return 'glances' in cmdline.lower()
#         except Exception:
#             return False
#     return False
#
#
# DJANGO_START_TIME = time.time()
#
#
# @login_required
# def dashboard_view(request):
#     selected_script_type = 'ffmpeg'
#     selected_det_set = 'auto'
#
#     if request.method == 'POST':
#         if 'start_glances' in request.POST:
#             start_glances()
#             time.sleep(1.5)
#         elif 'stop_glances' in request.POST:
#             stop_glances()
#         elif 'restart_glances' in request.POST:
#             stop_glances()
#             start_glances()
#             time.sleep(1.5)
#         if 'run_button' in request.POST:
#             selected_script_type = request.POST.get('script_type', 'opencv')
#             selected_det_set = request.POST.get('det_set', 'auto')
#             run_script_by_type(selected_script_type, selected_det_set)
#         elif 'stop_button' in request.POST:
#             stop_running_scripts()
#         return HttpResponseRedirect(reverse('dashboard:dashboard'))
#
#     running_info = get_running_script_info()
#     context = {
#         'running_info': running_info,
#         'django_uptime': time.time() - DJANGO_START_TIME,
#         'selected_script_type': running_info.get('type', 'opencv'),
#         'selected_det_set': running_info.get('det_set', 'auto'),
#         'det_set_options': [
#             'auto', '320,320', '480,480', '640,640', '800,800',
#             '1024,1024', '1280,1280', '1600,1600', '1920,1920', '2048,2048'
#         ],
#         'glances_running': is_glances_running(),
#     }
#     return render(request, 'dashboard/dashboard.html', context)
#
#
# def system_stats_view(request):
#     stats = get_system_stats()
#     stats['django_uptime'] = time.time() - DJANGO_START_TIME
#     stats['script_info'] = get_running_script_info()
#     return JsonResponse(stats)
