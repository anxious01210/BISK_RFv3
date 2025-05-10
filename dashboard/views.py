# dashboard/views.py
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseRedirect
from .scripts import run_script_by_type, stop_running_scripts, get_running_script_info
from .utils import get_system_stats
import time
import platform
from django.urls import reverse
from django.contrib.auth.decorators import login_required
import subprocess
import signal
import os

GLANCES_PID_FILE = '/tmp/glances_web.pid'


def start_glances():
    if not os.path.exists(GLANCES_PID_FILE):
        proc = subprocess.Popen(
            ['glances', '-w', '-B', '127.0.0.1', '-p', '61208'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid
        )
        with open(GLANCES_PID_FILE, 'w') as f:
            f.write(str(proc.pid))
        time.sleep(1.5)  # give it a moment to boot up


def stop_glances():
    if os.path.exists(GLANCES_PID_FILE):
        with open(GLANCES_PID_FILE, 'r') as f:
            pid = int(f.read().strip())
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        os.remove(GLANCES_PID_FILE)
    time.sleep(1.5)  # ensure time for system to reflect process termination


def is_glances_running():
    if os.path.exists(GLANCES_PID_FILE):
        try:
            with open(GLANCES_PID_FILE, 'r') as f:
                pid = int(f.read().strip())
            # Check if process exists
            os.kill(pid, 0)
            # Check if it's Glances
            cmdline = subprocess.check_output(['ps', '-p', str(pid), '-o', 'cmd=']).decode().strip()
            return 'glances' in cmdline.lower()
        except Exception:
            return False
    return False


DJANGO_START_TIME = time.time()


@login_required
def dashboard_view(request):
    selected_script_type = 'ffmpeg'
    selected_det_set = 'auto'

    if request.method == 'POST':
        if 'start_glances' in request.POST:
            start_glances()
            time.sleep(1.5)
        elif 'stop_glances' in request.POST:
            stop_glances()
        elif 'restart_glances' in request.POST:
            stop_glances()
            start_glances()
            time.sleep(1.5)
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
        'django_uptime': time.time() - DJANGO_START_TIME,
        'selected_script_type': running_info.get('type', 'opencv'),
        'selected_det_set': running_info.get('det_set', 'auto'),
        'det_set_options': [
            'auto', '320,320', '480,480', '640,640', '800,800',
            '1024,1024', '1280,1280', '1600,1600', '1920,1920', '2048,2048'
        ],
        'glances_running': is_glances_running(),
    }
    return render(request, 'dashboard/dashboard.html', context)


def system_stats_view(request):
    stats = get_system_stats()
    stats['django_uptime'] = time.time() - DJANGO_START_TIME
    stats['script_info'] = get_running_script_info()
    return JsonResponse(stats)
