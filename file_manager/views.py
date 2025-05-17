# file_manager/views.py
import os, json
from django.conf import settings
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import render
from django.http import JsonResponse
from .utils import safe_join
import mimetypes
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from django.views.decorators.csrf import csrf_exempt

def explorer_view(request):
    return render(request, 'file_manager/explorer.html')


def get_folder_size(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            try:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp):
                    total_size += os.path.getsize(fp)
            except Exception:
                pass  # Skip unreadable files
    return total_size


def list_folder_contents(request):
    # print("🛠️ RECURSIVE_FOLDER_SIZE setting:", getattr(settings, 'RECURSIVE_FOLDER_SIZE', False))
    rel_path = request.GET.get('path', '').strip('/')
    abs_path = safe_join(settings.MEDIA_ROOT, rel_path)

    folders = []
    files = []

    if os.path.exists(abs_path) and os.path.isdir(abs_path):
        for entry in os.listdir(abs_path):
            full = os.path.join(abs_path, entry)
            if os.path.isdir(full):
                size = get_folder_size(full) if getattr(settings, 'RECURSIVE_FOLDER_SIZE', False) else None
                # print(f"📁 Folder: {entry} — Size: {size}")
                folders.append({
                    "name": entry,
                    "size": size
                })
            else:
                try:
                    size = os.path.getsize(full)
                except Exception:
                    size = None
                mime_type, _ = mimetypes.guess_type(full)
                # ext = os.path.splitext(entry)[1].lower()
                ext = Path(entry).suffix.lower()
                files.append({
                    "name": entry,
                    "size": size,
                    "ext": ext,
                    "mime": mime_type or "application/octet-stream"
                })

    return JsonResponse({
        'current_path': rel_path,
        'folders': folders,
        'files': files,
    })


@csrf_exempt
def analyze_folders(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            paths = data.get("paths", [])

            total_files = 0
            total_folders = 0
            total_images = 0
            image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

            for rel_path in paths:
                abs_path = os.path.join(settings.MEDIA_ROOT, rel_path.lstrip("/"))
                if os.path.exists(abs_path):
                    for root, dirs, files in os.walk(abs_path):
                        total_folders += len(dirs)
                        total_files += len(files)
                        for f in files:
                            ext = os.path.splitext(f)[1].lower()
                            if ext in image_exts:
                                total_images += 1

            return JsonResponse({
                "total_files": total_files,
                "total_folders": total_folders,
                "total_images": total_images
            })

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=400)

    return JsonResponse({"error": "Invalid request method"}, status=405)






@csrf_exempt
def upload_files(request):
    if request.method != "POST":
        return JsonResponse({"status": "error", "message": "POST required"})

    path = request.POST.get("path", "").strip("/")
    mode = request.POST.get("folder_mode", "flat")
    saved = []

    try:
        relative_paths = json.loads(request.POST.get("relative_paths", "[]"))
    except json.JSONDecodeError:
        return JsonResponse({"status": "error", "message": "Invalid relative_paths"})

    print("📥 Upload endpoint triggered")
    print("POST:", request.POST)
    print("FILES:", request.FILES)

    files = request.FILES.getlist("files")
    if len(files) != len(relative_paths):
        return JsonResponse({"status": "error", "message": "Mismatch between files and relative paths"})

    for i, f in enumerate(files):
        if mode == "preserve":
            rel_path = relative_paths[i].strip("/")  # e.g., subfolder1/file.png
            save_path = os.path.join(path, rel_path)
        else:
            save_path = os.path.join(path, os.path.basename(f.name))

        full_path = os.path.join(settings.MEDIA_ROOT, save_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        with open(full_path, "wb+") as dest:
            for chunk in f.chunks():
                dest.write(chunk)

        saved.append(save_path)

    return JsonResponse({"status": "ok", "saved": saved})




# from pathlib import Path
#
# @csrf_exempt
# @require_POST
# def upload_file(request):
#     rel_path = request.POST.get("path", "").strip("/")
#     abs_path = safe_join(settings.MEDIA_ROOT, rel_path)
#
#     uploaded_files = request.FILES.getlist("files")
#     folder_mode = request.POST.get("folder_mode", "flat")  # "flat" or "preserve"
#
#     saved = []
#     for f in uploaded_files:
#         # Preserve full relative path if requested
#         if folder_mode == "preserve" and "/" in f.name:
#             relative_target = f.name  # e.g. sub1/img.png or sub1/sub2/doc.txt
#         else:
#             relative_target = os.path.basename(f.name)
#
#         target_path = os.path.join(abs_path, relative_target)
#         os.makedirs(os.path.dirname(target_path), exist_ok=True)
#
#         with open(target_path, "wb+") as dest:
#             for chunk in f.chunks():
#                 dest.write(chunk)
#
#         saved.append(relative_target)
#
#     return JsonResponse({"status": "ok", "saved": saved})