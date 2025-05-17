import os
from django.conf import settings
from django.shortcuts import render, redirect
from django.http import HttpResponseForbidden, HttpResponseRedirect, HttpResponse
from .utils import safe_join, is_image_file
from django.views.decorators.csrf import csrf_exempt
from django.contrib import messages
import json
import io
import zipfile
from django.http import StreamingHttpResponse
from django.http import JsonResponse

def browser_view(request):
    rel_path = request.GET.get('path', '')
    abs_path = safe_join(settings.MEDIA_ROOT, rel_path)

    if not os.path.exists(abs_path):
        return HttpResponseForbidden("Invalid path")

    files = []
    dirs = []

    for entry in os.listdir(abs_path):
        full_path = os.path.join(abs_path, entry)
        rel_entry_path = os.path.join(rel_path, entry)
        if os.path.isdir(full_path):
            dirs.append({'name': entry, 'path': rel_entry_path})
        else:
            files.append({
                'name': entry,
                'path': rel_entry_path,
                'is_image': is_image_file(entry)
            })

    context = {
        'current_path': rel_path,
        'parent_path': os.path.dirname(rel_path),
        'dirs': sorted(dirs, key=lambda d: d['name']),
        'files': sorted(files, key=lambda f: f['name']),
    }
    return render(request, 'media_manager/browser.html', context)


@csrf_exempt
def upload_files(request):
    print("🎯 Upload view triggered")  # <-- DEBUG LOG
    print("request.FILES keys:", request.FILES.keys())
    print("request.POST:", request.POST)
    if request.method != 'POST':
        return HttpResponseForbidden("Only POST allowed")

    rel_path = request.POST.get('target_path', '')
    abs_path = safe_join(settings.MEDIA_ROOT, rel_path)

    if not os.path.exists(abs_path):
        os.makedirs(abs_path)

    # uploaded_files = request.FILES.getlist('files[]')
    uploaded_files = request.FILES.getlist('files')
    print("Uploaded count:", len(uploaded_files))
    saved_count = 0

    for f in uploaded_files:
        file_path = os.path.join(abs_path, f.name)
        with open(file_path, 'wb+') as dest:
            for chunk in f.chunks():
                dest.write(chunk)
        saved_count += 1

    messages.success(request, f"✅ Uploaded {saved_count} file(s) to /media/{rel_path}/")
    return HttpResponseRedirect(f"/media-manager/?path={rel_path}")
    # return HttpResponse("Upload finished")


def create_folder(request):
    pass


def delete_files(request):
    if request.method != 'POST':
        return HttpResponseForbidden("Only POST allowed")

    rel_path = request.POST.get('target_path', '')
    abs_dir = safe_join(settings.MEDIA_ROOT, rel_path)

    try:
        selected_files = json.loads(request.POST.get('selected_files', '[]'))
    except json.JSONDecodeError:
        selected_files = []

    deleted = 0
    for rel_file in selected_files:
        try:
            abs_file = safe_join(settings.MEDIA_ROOT, rel_file)
            if os.path.exists(abs_file) and abs_file.startswith(abs_dir):
                os.remove(abs_file)
                deleted += 1
        except Exception as e:
            print(f"Error deleting {rel_file}: {e}")

    messages.success(request, f"🗑 Deleted {deleted} file(s).")
    return HttpResponseRedirect(f"/media-manager/?path={rel_path}")


def rename_file(request):
    pass


def move_files(request):
    if request.method != 'POST':
        return HttpResponseForbidden("Only POST allowed")

    rel_path = request.POST.get('target_path', '')
    abs_dir = safe_join(settings.MEDIA_ROOT, rel_path)

    # destination = request.POST.get('destination', '').strip().lstrip('/')
    # abs_dest = safe_join(settings.MEDIA_ROOT, destination)
    destination = request.POST.get('destination', '').strip().strip('/')
    if destination:
        abs_dest = safe_join(settings.MEDIA_ROOT, destination)
    else:
        abs_dest = settings.MEDIA_ROOT

    os.makedirs(abs_dest, exist_ok=True)

    try:
        selected_files = json.loads(request.POST.get('selected_files', '[]'))
    except json.JSONDecodeError:
        selected_files = []

    moved = 0
    for rel_file in selected_files:
        abs_file = safe_join(settings.MEDIA_ROOT, rel_file)
        if os.path.exists(abs_file):
            new_path = os.path.join(abs_dest, os.path.basename(abs_file))
            os.rename(abs_file, new_path)
            moved += 1

    # messages.success(request, f"📁 Moved {moved} file(s) to /media/{destination}/")
    messages.success(request, f"📁 Moved {moved} file(s) to /media/{destination or ''}")

    # return HttpResponseRedirect(f"/media-manager/?path={rel_path}")
    return HttpResponseRedirect(f"/media-manager/?path={rel_path}&moved={moved}&dest={destination}")






def download_files(request):
    if request.method != 'POST':
        return HttpResponseForbidden("Only POST allowed")

    rel_path = request.POST.get('target_path', '')
    try:
        selected_files = json.loads(request.POST.get('selected_files', '[]'))
    except json.JSONDecodeError:
        selected_files = []

    if not selected_files:
        return HttpResponseRedirect(f"/media-manager/?path={rel_path}")

    # Prepare in-memory ZIP
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for rel_file in selected_files:
            abs_file = safe_join(settings.MEDIA_ROOT, rel_file)
            if os.path.exists(abs_file):
                zip_file.write(abs_file, arcname=os.path.basename(abs_file))

    zip_buffer.seek(0)

    response = StreamingHttpResponse(zip_buffer, content_type='application/zip')
    response['Content-Disposition'] = 'attachment; filename=selected_files.zip'
    return response



def folder_tree(request):
    folder_list = []

    for root, dirs, files in os.walk(settings.MEDIA_ROOT):
        rel_root = os.path.relpath(root, settings.MEDIA_ROOT)
        if rel_root == '.':
            rel_root = ''
        for d in dirs:
            full_path = os.path.join(rel_root, d).replace("\\", "/")  # Ensure forward slashes
            folder_list.append({
                "path": full_path,
                "name": d
            })

    return JsonResponse(folder_list, safe=False)
