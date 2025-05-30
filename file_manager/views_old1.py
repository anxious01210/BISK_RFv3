# file_manager/views.py
import os
import uuid
import json
import threading
import subprocess
from django.conf import settings
from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse, HttpResponseNotAllowed
from .utils import safe_join
import mimetypes
from django.views.decorators.csrf import csrf_exempt
from pathlib import Path
from django.views.decorators.http import require_POST
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from extras.embedding_utils import run_embedding_on_paths
from urllib.parse import unquote
import time
BASE_DIR = settings.BASE_DIR
from .script_runner import running_processes, job_output_buffers

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

    # print("📥 Upload endpoint triggered")
    # print("POST:", request.POST)
    # print("FILES:", request.FILES)

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




@csrf_exempt
@require_POST
def run_embeddings_script(request):
    try:
        data = json.loads(request.body)
        rel_paths = data.get("paths", [])
        det_set = data.get("det_set", "auto")
        force = data.get("force", False)
        print("📥 Received paths:", rel_paths)
        abs_paths = []
        for rel in rel_paths:
            rel = unquote(rel)
            rel = rel.lstrip("/")  # remove any leading /
            full_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, rel))
            # print("🔍 Checking:", full_path)
            print(f"🧪 Raw rel = {rel}")
            print(f"📁 full_path = {full_path}")
            print(f"📄 exists? {os.path.exists(full_path)}")
            if os.path.exists(full_path):
                abs_paths.append(full_path)
        print("✅ Valid absolute paths:", abs_paths)
        if not abs_paths:
            return JsonResponse({"error": "No valid files or folders found."}, status=400)
        print("✅ Final valid abs_paths:", abs_paths)
        result = run_embedding_on_paths(paths=abs_paths, det_set=det_set, force=force)
        summary = (
            f"✅ Saved: {result['saved']}, "
            f"⚠️ Skipped: {result['skipped']}, "
            f"📦 PKL: {os.path.basename(result['pkl'])}"
        )
        return JsonResponse({"message": summary})
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)







running_processes = {}
job_output_buffers = {}

def run_sort_faces_script(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            paths = data.get("paths", [])
            det_sets = data.get("det_sets", [])
            options = data.get("options", {})

            print("📥 Received POST /run-sort-faces/")
            print("   ➕ Paths:", paths)
            print("   📐 Det_sets:", det_sets)
            print("   ⚙️ Options:", options)

            if not paths or not det_sets:
                print("❌ Missing paths or det_sets")
                return JsonResponse({"error": "Missing parameters"}, status=400)

            # Only support single path input for now
            input_path = os.path.join(settings.MEDIA_ROOT, paths[0])
            det_set_arg = ",".join(det_sets)

            command = [
                "python3",
                "extras/sort_faces_by_detset.py",
                "--input", input_path,
                "--det_sets", det_set_arg,
                "--terminal_logs"
            ]

            job_id = str(uuid.uuid4())
            print(f"🚀 Generated job_id: {job_id}")
            output_buffer = []
            job_output_buffers[job_id] = output_buffer

            def stream_output():
                print(f"🎬 Running subprocess for job {job_id} with command:")
                print("   ", " ".join(command))

                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=1,
                    text=True
                )
                running_processes[job_id] = process

                for line in process.stdout:
                    output_buffer.append(line)
                    print(f"[{job_id}] {line.strip()}")

                process.wait()
                print(f"✅ Job {job_id} completed.")
                running_processes.pop(job_id, None)

            threading.Thread(target=stream_output, daemon=True).start()

            return JsonResponse({"job_id": job_id})

        except Exception as e:
            print("❌ Exception in run_sort_faces_script:", str(e))
            return JsonResponse({"error": "Internal server error"}, status=500)
    else:
        return JsonResponse({"error": "Invalid method"}, status=405)

@csrf_exempt
def stop_sort_faces_job(request, job_id):
    process = running_processes.get(job_id)
    if process:
        process.terminate()
        return JsonResponse({"status": "terminated"})
    return JsonResponse({"status": "not_found"}, status=404)

def stream_sort_faces_output(request, job_id):
    def event_stream():
        sent = 0
        while True:
            buffer = job_output_buffers.get(job_id, [])
            if sent < len(buffer):
                for line in buffer[sent:]:
                    yield f"data: {line}\n\n"
                sent = len(buffer)
            if job_id not in running_processes:
                break
    return StreamingHttpResponse(event_stream(), content_type='text/event-stream')










# running_processes = {}
# process_outputs = {}
# process_completed = {}
#
# def stream_log_output(job_id):
#     def event_stream():
#         buffer = process_outputs.get(job_id, [])
#         idx = 0
#         while True:
#             if job_id not in running_processes and not buffer[idx:]:
#                 break
#             new_lines = buffer[idx:]
#             idx = len(buffer)
#             for line in new_lines:
#                 yield f"data: {line}\n\n"
#             time.sleep(0.4)
#     return StreamingHttpResponse(event_stream(), content_type="text/event-stream")
#
# @csrf_exempt
# @require_POST
# def run_sort_faces_script(request):
#     data = json.loads(request.body)
#     rel_paths = data.get("paths", [])
#     det_sets = data.get("det_sets", [])
#     options = data.get("options", {})
#
#     abs_paths = []
#     for rel in rel_paths:
#         rel = rel.lstrip("/")
#         full_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, rel))
#         abs_paths.append(full_path)
#
#     input_folder = abs_paths[0]  # Only one for now
#     job_id = str(uuid.uuid4())
#     print(f"[DEBUG] 🏷️ Generated job_id = {job_id}")
#
#     cmd = ["python3", "extras/sort_faces_by_detset.py", "--input", input_folder]
#     if det_sets:
#         cmd += ["--det_sets", ",".join(det_sets)]
#     if options.get("ENABLE_TERMINAL_LOGS"):
#         cmd += ["--terminal_logs"]
#
#     env = os.environ.copy()
#     output_buffer = []
#     process_outputs[job_id] = output_buffer
#
#     def run():
#         try:
#             print(f"[DEBUG] 🧵 Running script: {' '.join(cmd)}")
#             process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
#             running_processes[job_id] = process
#             for line in iter(process.stdout.readline, ''):
#                 clean = line.strip()
#                 print(f"📤 [DEBUG] Script output: {clean}")
#                 output_buffer.append(clean)
#             process.wait()
#             print(f"[DEBUG] 🟢 Script for job {job_id} fully finished.")
#         finally:
#             print(f"[DEBUG] 🧼 Cleaning up completed job {job_id}")
#             process_completed[job_id] = True
#             process_outputs.pop(job_id, None)
#             running_processes.pop(job_id, None)
#
#     thread = threading.Thread(target=run)
#     thread.start()
#
#     return JsonResponse({"job_id": job_id})
#
# def stream_sort_faces_output(request, job_id):
#     return stream_log_output(job_id)
#
# @csrf_exempt
# @require_POST
# def stop_sort_faces_job(request, job_id):
#     print(f"[DEBUG] 🔌 stop_sort_faces_job called with job_id = {job_id}")
#     print(f"[DEBUG] Available job_ids in running_processes: {list(running_processes.keys())}")
#
#     if job_id in running_processes:
#         process = running_processes[job_id]
#         if process is not None:
#             print(f"[DEBUG] 🛑 Terminating running process {job_id}")
#             process.terminate()
#             running_processes.pop(job_id, None)
#             return JsonResponse({"status": "terminated"})
#         else:
#             print(f"[DEBUG] ⚠️ Process is None for {job_id} — may already be cleaned up")
#
#     if process_completed.get(job_id):
#         print(f"[DEBUG] ✅ Job {job_id} was already completed.")
#         return JsonResponse({"status": "already completed"})
#
#     print(f"[DEBUG] ❌ Job {job_id} not found.")
#     return JsonResponse({"error": "No running process found."}, status=404)
#
# from django.http import StreamingHttpResponse
# import os
# import time
#
# LOG_DIR = os.path.join(BASE_DIR, "media/logs/sort_faces/")
#
# def stream_sort_faces_logs(request, job_id):
#     def log_stream():
#         log_path = os.path.join(LOG_DIR, f"{job_id}.log")
#         last_pos = 0
#         timeout = time.time() + 60  # 1-minute timeout
#
#         while time.time() < timeout:
#             if os.path.exists(log_path):
#                 with open(log_path, "r") as f:
#                     f.seek(last_pos)
#                     new_lines = f.readlines()
#                     if new_lines:
#                         for line in new_lines:
#                             yield f"data: {line.strip()}\n\n"
#                         last_pos = f.tell()
#             time.sleep(0.5)
#
#         yield "event: close\ndata: end\n\n"
#
#     return StreamingHttpResponse(log_stream(), content_type='text/event-stream')










# # file_manager/views.py
# import os
# import uuid
# import json
# import threading
# import subprocess
# from django.conf import settings
# from django.shortcuts import render
# from django.http import JsonResponse, StreamingHttpResponse, HttpResponseNotAllowed, HttpResponseNotFound
# from .utils import safe_join
# import mimetypes
# from django.views.decorators.csrf import csrf_exempt
# from pathlib import Path
# from django.views.decorators.http import require_POST
# from django.core.files.storage import default_storage
# from django.core.files.base import ContentFile
# from extras.embedding_utils import run_embedding_on_paths
# from urllib.parse import unquote
# import time
# BASE_DIR = settings.BASE_DIR
#
# # Adjust this path as needed
# # LOG_DIR = os.path.join(BASE_DIR, "media/logs/sort_faces/")
# # def stream_sort_faces_logs(request, job_id):
# #     def log_stream():
# #         log_path = os.path.join(LOG_DIR, f"{job_id}.log")
# #         last_pos = 0
# #         timeout = time.time() + 60  # 1-minute timeout
# #
# #         while time.time() < timeout:
# #             if os.path.exists(log_path):
# #                 with open(log_path, "r") as f:
# #                     f.seek(last_pos)
# #                     new_lines = f.readlines()
# #                     if new_lines:
# #                         for line in new_lines:
# #                             yield f"data: {line.strip()}\n\n"
# #                         last_pos = f.tell()
# #             time.sleep(0.5)
# #
# #         yield "event: close\ndata: end\n\n"
# #
# #     return StreamingHttpResponse(log_stream(), content_type='text/event-stream')
#
#
# def explorer_view(request):
#     return render(request, 'file_manager/explorer.html')
#
#
# def get_folder_size(path):
#     total_size = 0
#     for dirpath, _, filenames in os.walk(path):
#         for f in filenames:
#             try:
#                 fp = os.path.join(dirpath, f)
#                 if os.path.isfile(fp):
#                     total_size += os.path.getsize(fp)
#             except Exception:
#                 pass  # Skip unreadable files
#     return total_size
#
#
# def list_folder_contents(request):
#     # print("🛠️ RECURSIVE_FOLDER_SIZE setting:", getattr(settings, 'RECURSIVE_FOLDER_SIZE', False))
#     rel_path = request.GET.get('path', '').strip('/')
#     abs_path = safe_join(settings.MEDIA_ROOT, rel_path)
#
#     folders = []
#     files = []
#
#     if os.path.exists(abs_path) and os.path.isdir(abs_path):
#         for entry in os.listdir(abs_path):
#             full = os.path.join(abs_path, entry)
#             if os.path.isdir(full):
#                 size = get_folder_size(full) if getattr(settings, 'RECURSIVE_FOLDER_SIZE', False) else None
#                 # print(f"📁 Folder: {entry} — Size: {size}")
#                 folders.append({
#                     "name": entry,
#                     "size": size
#                 })
#             else:
#                 try:
#                     size = os.path.getsize(full)
#                 except Exception:
#                     size = None
#                 mime_type, _ = mimetypes.guess_type(full)
#                 # ext = os.path.splitext(entry)[1].lower()
#                 ext = Path(entry).suffix.lower()
#                 files.append({
#                     "name": entry,
#                     "size": size,
#                     "ext": ext,
#                     "mime": mime_type or "application/octet-stream"
#                 })
#
#     return JsonResponse({
#         'current_path': rel_path,
#         'folders': folders,
#         'files': files,
#     })
#
#
# @csrf_exempt
# def analyze_folders(request):
#     if request.method == "POST":
#         try:
#             data = json.loads(request.body)
#             paths = data.get("paths", [])
#
#             total_files = 0
#             total_folders = 0
#             total_images = 0
#             image_exts = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
#
#             for rel_path in paths:
#                 abs_path = os.path.join(settings.MEDIA_ROOT, rel_path.lstrip("/"))
#                 if os.path.exists(abs_path):
#                     for root, dirs, files in os.walk(abs_path):
#                         total_folders += len(dirs)
#                         total_files += len(files)
#                         for f in files:
#                             ext = os.path.splitext(f)[1].lower()
#                             if ext in image_exts:
#                                 total_images += 1
#
#             return JsonResponse({
#                 "total_files": total_files,
#                 "total_folders": total_folders,
#                 "total_images": total_images
#             })
#
#         except Exception as e:
#             return JsonResponse({"error": str(e)}, status=400)
#
#     return JsonResponse({"error": "Invalid request method"}, status=405)
#
#
#
#
#
#
# @csrf_exempt
# def upload_files(request):
#     if request.method != "POST":
#         return JsonResponse({"status": "error", "message": "POST required"})
#
#     path = request.POST.get("path", "").strip("/")
#     mode = request.POST.get("folder_mode", "flat")
#     saved = []
#
#     try:
#         relative_paths = json.loads(request.POST.get("relative_paths", "[]"))
#     except json.JSONDecodeError:
#         return JsonResponse({"status": "error", "message": "Invalid relative_paths"})
#
#     # print("📥 Upload endpoint triggered")
#     # print("POST:", request.POST)
#     # print("FILES:", request.FILES)
#
#     files = request.FILES.getlist("files")
#     if len(files) != len(relative_paths):
#         return JsonResponse({"status": "error", "message": "Mismatch between files and relative paths"})
#
#     for i, f in enumerate(files):
#         if mode == "preserve":
#             rel_path = relative_paths[i].strip("/")  # e.g., subfolder1/file.png
#             save_path = os.path.join(path, rel_path)
#         else:
#             save_path = os.path.join(path, os.path.basename(f.name))
#
#         full_path = os.path.join(settings.MEDIA_ROOT, save_path)
#         os.makedirs(os.path.dirname(full_path), exist_ok=True)
#
#         with open(full_path, "wb+") as dest:
#             for chunk in f.chunks():
#                 dest.write(chunk)
#
#         saved.append(save_path)
#
#     return JsonResponse({"status": "ok", "saved": saved})
#
#
#
#
# @csrf_exempt
# @require_POST
# def run_embeddings_script(request):
#     try:
#         data = json.loads(request.body)
#         rel_paths = data.get("paths", [])
#         det_set = data.get("det_set", "auto")
#         force = data.get("force", False)
#         print("📥 Received paths:", rel_paths)
#         abs_paths = []
#         for rel in rel_paths:
#             rel = unquote(rel)
#             rel = rel.lstrip("/")  # remove any leading /
#             full_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, rel))
#             # print("🔍 Checking:", full_path)
#             print(f"🧪 Raw rel = {rel}")
#             print(f"📁 full_path = {full_path}")
#             print(f"📄 exists? {os.path.exists(full_path)}")
#             if os.path.exists(full_path):
#                 abs_paths.append(full_path)
#         print("✅ Valid absolute paths:", abs_paths)
#         if not abs_paths:
#             return JsonResponse({"error": "No valid files or folders found."}, status=400)
#         print("✅ Final valid abs_paths:", abs_paths)
#         result = run_embedding_on_paths(paths=abs_paths, det_set=det_set, force=force)
#         summary = (
#             f"✅ Saved: {result['saved']}, "
#             f"⚠️ Skipped: {result['skipped']}, "
#             f"📦 PKL: {os.path.basename(result['pkl'])}"
#         )
#         return JsonResponse({"message": summary})
#     except Exception as e:
#         return JsonResponse({"error": str(e)}, status=500)
#
#
#
# running_processes = {}
#
# @csrf_exempt
# def run_sort_faces(request):
#     if request.method != "POST":
#         return JsonResponse({"error": "POST required"}, status=400)
#
#     import json
#     data = json.loads(request.body)
#     paths = data.get("paths", [])
#     det_sets = data.get("det_sets", [])
#     options = data.get("options", {})
#
#     if not paths:
#         return JsonResponse({"error": "No paths provided"}, status=400)
#
#     job_id = str(uuid.uuid4())
#     input_path = os.path.join(settings.MEDIA_ROOT, paths[0])
#     det_sets_arg = ",".join(det_sets)
#
#     cmd = [
#         "python3", "extras/sort_faces_by_detset.py",
#         "--input", input_path,
#         "--det_sets", det_sets_arg,
#         "--terminal_logs"
#     ]
#
#     for key, value in options.items():
#         if isinstance(value, bool):
#             if value:
#                 cmd.append(f"--{key}")
#         else:
#             cmd.extend([f"--{key}", str(value)])
#
#     print(f"[DEBUG] 🧵 Running script: {' '.join(cmd)}")
#
#     def run_script():
#         process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         running_processes[job_id] = process
#         stdout, stderr = process.communicate()
#         print(stdout.decode())
#         print(stderr.decode())
#         running_processes.pop(job_id, None)
#
#     thread = threading.Thread(target=run_script)
#     thread.start()
#
#     return JsonResponse({"job_id": job_id})
#
# @csrf_exempt
# def stop_sort_faces(request, job_id):
#     process = running_processes.get(job_id)
#     if process:
#         process.terminate()
#         running_processes.pop(job_id, None)
#         return JsonResponse({"stopped": True})
#     return HttpResponseNotFound("Job not found")



# running_processes = {}
# process_outputs = {}
# process_completed = {}
#
# def stream_log_output(job_id):
#     def event_stream():
#         buffer = process_outputs.get(job_id, [])
#         idx = 0
#         while True:
#             if job_id not in running_processes and not buffer[idx:]:
#                 break
#             new_lines = buffer[idx:]
#             idx = len(buffer)
#             for line in new_lines:
#                 yield f"data: {line}\n\n"
#             time.sleep(0.4)
#     return StreamingHttpResponse(event_stream(), content_type="text/event-stream")
#
# @csrf_exempt
# @require_POST
# def run_sort_faces_script(request):
#     data = json.loads(request.body)
#     rel_paths = data.get("paths", [])
#     det_sets = data.get("det_sets", [])
#     options = data.get("options", {})
#
#     abs_paths = []
#     for rel in rel_paths:
#         rel = rel.lstrip("/")
#         full_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, rel))
#         abs_paths.append(full_path)
#
#     input_folder = abs_paths[0]  # Only one for now
#     job_id = str(uuid.uuid4())
#     print(f"[DEBUG] 🏷️ Generated job_id = {job_id}")
#
#     cmd = ["python3", "extras/sort_faces_by_detset.py", "--input", input_folder]
#     if det_sets:
#         cmd += ["--det_sets", ",".join(det_sets)]
#     if options.get("ENABLE_TERMINAL_LOGS"):
#         cmd += ["--terminal_logs"]
#
#     env = os.environ.copy()
#     output_buffer = []
#     process_outputs[job_id] = output_buffer
#
#     def run():
#         try:
#             print(f"[DEBUG] 🧵 Running script: {' '.join(cmd)}")
#             process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
#             running_processes[job_id] = process
#             for line in iter(process.stdout.readline, ''):
#                 clean = line.strip()
#                 print(f"📤 [DEBUG] Script output: {clean}")
#                 output_buffer.append(clean)
#             process.wait()
#             print(f"[DEBUG] 🟢 Script for job {job_id} fully finished.")
#         finally:
#             print(f"[DEBUG] 🧼 Cleaning up completed job {job_id}")
#             process_completed[job_id] = True
#             process_outputs.pop(job_id, None)
#             running_processes.pop(job_id, None)
#
#     thread = threading.Thread(target=run)
#     thread.start()
#
#     return JsonResponse({"job_id": job_id})
#
# def stream_sort_faces_output(request, job_id):
#     def event_stream():
#         buffer = job_output_buffers.get(job_id)
#         if not buffer:
#             yield "⛔ Job not found.\n"
#             return
#         while True:
#             line = buffer.readline(timeout=1.0)  # custom method you define
#             if line is None:
#                 break
#             yield line
#     return StreamingHttpResponse(event_stream(), content_type='text/plain')
#
# @csrf_exempt
# @require_POST
# def stop_sort_faces_job(request, job_id):
#     print(f"[DEBUG] 🔌 stop_sort_faces_job called with job_id = {job_id}")
#     print(f"[DEBUG] Available job_ids in running_processes: {list(running_processes.keys())}")
#
#     if job_id in running_processes:
#         process = running_processes[job_id]
#         if process is not None:
#             print(f"[DEBUG] 🛑 Terminating running process {job_id}")
#             process.terminate()
#             running_processes.pop(job_id, None)
#             return JsonResponse({"status": "terminated"})
#         else:
#             print(f"[DEBUG] ⚠️ Process is None for {job_id} — may already be cleaned up")
#
#     if process_completed.get(job_id):
#         print(f"[DEBUG] ✅ Job {job_id} was already completed.")
#         return JsonResponse({"status": "already completed"})
#
#     print(f"[DEBUG] ❌ Job {job_id} not found.")
#     return JsonResponse({"error": "No running process found."}, status=404)
#
# from django.http import StreamingHttpResponse
# import os
# import time
#
# LOG_DIR = os.path.join(BASE_DIR, "media/logs/sort_faces/")
#
# from .script_runner import job_output_buffers
#
# def stream_sort_faces_logs(request, job_id):
#     job_id_str = str(job_id)
#
#     def event_stream():
#         sent_index = 0
#         while True:
#             logs = job_output_buffers.get(job_id_str, [])
#             if sent_index < len(logs):
#                 for line in logs[sent_index:]:
#                     yield f"data: {line}\n\n"
#                 sent_index = len(logs)
#             time.sleep(0.5)
#
#     return StreamingHttpResponse(event_stream(), content_type='text/event-stream')

# def stream_sort_faces_logs(request, job_id):
#     def log_stream():
#         log_path = os.path.join(LOG_DIR, f"{job_id}.log")
#         last_pos = 0
#         timeout = time.time() + 60  # 1-minute timeout
#
#         while time.time() < timeout:
#             if os.path.exists(log_path):
#                 with open(log_path, "r") as f:
#                     f.seek(last_pos)
#                     new_lines = f.readlines()
#                     if new_lines:
#                         for line in new_lines:
#                             yield f"data: {line.strip()}\n\n"
#                         last_pos = f.tell()
#             time.sleep(0.5)
#
#         yield "event: close\ndata: end\n\n"
#
#     return StreamingHttpResponse(log_stream(), content_type='text/event-stream')







# running_processes = {}
# process_outputs = {}
# process_completed = {}
#
# def stream_log_output(job_id):
#     def event_stream():
#         buffer = process_outputs.get(job_id, [])
#         idx = 0
#         while True:
#             if job_id not in running_processes and not buffer[idx:]:
#                 break
#             new_lines = buffer[idx:]
#             idx = len(buffer)
#             for line in new_lines:
#                 yield f"data: {line}\n\n"
#             time.sleep(0.4)
#     return StreamingHttpResponse(event_stream(), content_type="text/event-stream")
#
# @csrf_exempt
# @require_POST
# def run_sort_faces_script(request):
#     data = json.loads(request.body)
#     rel_paths = data.get("paths", [])
#     det_sets = data.get("det_sets", [])
#     options = data.get("options", {})
#
#     abs_paths = []
#     for rel in rel_paths:
#         rel = rel.lstrip("/")
#         full_path = os.path.normpath(os.path.join(settings.MEDIA_ROOT, rel))
#         abs_paths.append(full_path)
#
#     input_folder = abs_paths[0]  # Only one for now
#     job_id = str(uuid.uuid4())
#     print(f"[DEBUG] 🏷️ Generated job_id = {job_id}")
#
#     cmd = ["python3", "extras/sort_faces_by_detset.py", "--input", input_folder]
#     if det_sets:
#         cmd += ["--det_sets", ",".join(det_sets)]
#     if options.get("ENABLE_TERMINAL_LOGS"):
#         cmd += ["--terminal_logs"]
#
#     env = os.environ.copy()
#     output_buffer = []
#     process_outputs[job_id] = output_buffer
#
#     def run():
#         try:
#             print(f"[DEBUG] 🧵 Running script: {' '.join(cmd)}")
#             process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, text=True)
#             running_processes[job_id] = process
#             for line in iter(process.stdout.readline, ''):
#                 clean = line.strip()
#                 print(f"📤 [DEBUG] Script output: {clean}")
#                 output_buffer.append(clean)
#             process.wait()
#             print(f"[DEBUG] 🟢 Script for job {job_id} fully finished.")
#         finally:
#             print(f"[DEBUG] 🧼 Cleaning up completed job {job_id}")
#             process_completed[job_id] = True
#             process_outputs.pop(job_id, None)
#             running_processes.pop(job_id, None)
#
#     thread = threading.Thread(target=run)
#     thread.start()
#
#     return JsonResponse({"job_id": job_id})
#
# def stream_sort_faces_output(request, job_id):
#     return stream_log_output(job_id)
#
# @csrf_exempt
# @require_POST
# def stop_sort_faces_job(request, job_id):
#     print(f"[DEBUG] 🔌 stop_sort_faces_job called with job_id = {job_id}")
#     print(f"[DEBUG] Available job_ids in running_processes: {list(running_processes.keys())}")
#
#     if job_id in running_processes:
#         process = running_processes[job_id]
#         if process is not None:
#             print(f"[DEBUG] 🛑 Terminating running process {job_id}")
#             process.terminate()
#             running_processes.pop(job_id, None)
#             return JsonResponse({"status": "terminated"})
#         else:
#             print(f"[DEBUG] ⚠️ Process is None for {job_id} — may already be cleaned up")
#
#     if process_completed.get(job_id):
#         print(f"[DEBUG] ✅ Job {job_id} was already completed.")
#         return JsonResponse({"status": "already completed"})
#
#     print(f"[DEBUG] ❌ Job {job_id} not found.")
#     return JsonResponse({"error": "No running process found."}, status=404)