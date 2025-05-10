# dashboard/views.py
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseRedirect
from .scripts import run_script_by_type, stop_running_scripts, get_running_script_info
from .utils import get_system_stats
import time
import platform
from django.urls import reverse

DJANGO_START_TIME = time.time()

def dashboard_view(request):
    selected_script_type = 'opencv'
    selected_det_set = 'auto'

    if request.method == 'POST':
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
        ]
    }
    return render(request, 'dashboard/dashboard.html', context)

def system_stats_view(request):
    stats = get_system_stats()
    stats['django_uptime'] = time.time() - DJANGO_START_TIME
    stats['script_info'] = get_running_script_info()
    return JsonResponse(stats)







# # dashboard/views.py
# from django.shortcuts import render
# from django.http import JsonResponse, HttpResponseRedirect
# from .scripts import run_script_by_type, stop_running_scripts, get_running_script_info
# from .utils import get_system_stats
# import time
# import platform
# from django.urls import reverse
#
# DJANGO_START_TIME = time.time()
#
# def dashboard_view(request):
#     selected_script_type = 'opencv'
#     selected_det_set = 'auto'
#
#     if request.method == 'POST':
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
#     }
#     return render(request, 'dashboard/dashboard.html', context)
#
# def system_stats_view(request):
#     stats = get_system_stats()
#     stats['django_uptime'] = time.time() - DJANGO_START_TIME
#     stats['script_info'] = get_running_script_info()
#     return JsonResponse(stats)


# # dashboard/views.py
# from django.shortcuts import render
# from django.http import JsonResponse
# from .scripts import run_script_by_type
# from .utils import get_system_stats
#
# def dashboard_view(request):
#     if request.method == 'POST':
#         script_type = request.POST.get('script_type')  # now using string: 'opencv' or 'ffmpeg'
#         det_set = request.POST.get('det_set')
#         run_script_by_type(script_type, det_set)
#     return render(request, 'dashboard/dashboard.html')
#
# def system_stats_view(request):
#     stats = get_system_stats()
#     return JsonResponse(stats)







# # dashboard/views.py
# from django.shortcuts import render
# from django.http import JsonResponse
# from .scripts import run_script_by_type
# from .utils import get_system_stats
#
# def dashboard_view(request):
#     if request.method == 'POST':
#         script_type = int(request.POST.get('script_type'))
#         det_set = request.POST.get('det_set')
#         run_script_by_type(script_type, det_set)
#     return render(request, 'dashboard/dashboard.html')
#
# def system_stats_view(request):
#     stats = get_system_stats()
#     return JsonResponse(stats)