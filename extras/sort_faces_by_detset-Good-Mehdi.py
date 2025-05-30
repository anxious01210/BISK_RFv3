# ** Mehdi _ A very very very good one, but it lacks the auto GPU memory check and act mechanism.
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from insightface.app import FaceAnalysis
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown

# === Configuration ===
DETECTION_SIZES = ["2048x2048", "1600x1600", "1280x1280", "1024x1024", "896x896", "800x800", "768x768", "704x704", "640x640"]
ENABLE_PREVIEW_IMAGE = True
ENABLE_PREVIEW_OVERLAY = True
PREVIEW_CROP_MARGIN_PERCENT = 30  # Set to None to disable cropping
ENABLE_TERMINAL_LOGS = True
OUTPUT_UNDER_MEDIA_FOLDER = True
MAX_GPU_MEMORY_PERCENT = 95
MEMORY_CHECK_INTERVAL = 10

# === Preview Text Styling Settings ===
PREVIEW_TEXT_COLOR = (255, 255, 255)
PREVIEW_TEXT_SIZE = 0.5
PREVIEW_TEXT_BOLD = True
PREVIEW_TEXT_BG_COLOR = (0, 0, 0)
PREVIEW_TEXT_BG_OPACITY = 0.6
ENABLE_TEXT_BG = True
ENABLE_CUSTOM_FONT = True

parser = argparse.ArgumentParser()
parser.add_argument("--input", type=str, required=True, help="Input folder of student images")
args = parser.parse_args()

INPUT_FOLDER = args.input
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
OUTPUT_BASE = os.path.join("media", f"Sorted faces ({TIMESTAMP})") if OUTPUT_UNDER_MEDIA_FOLDER else os.path.join(INPUT_FOLDER, f"Sorted faces ({TIMESTAMP})")
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "previews"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_BASE, "bad"), exist_ok=True)

nvmlInit()
handle = nvmlDeviceGetHandleByIndex(0)
def get_free_gpu_percent():
    mem = nvmlDeviceGetMemoryInfo(handle)
    return (mem.free / mem.total) * 100

detectors = {}
for size in DETECTION_SIZES:
    w, h = map(int, size.split("x"))
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0, det_size=(w, h))
    detectors[size] = app

image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
log_path = os.path.join(OUTPUT_BASE, "sorted_face_log.csv")
summary = {}

def crop_with_margin(image, box, margin_percent):
    if margin_percent is None:
        return image
    x1, y1, x2, y2 = box
    h, w = image.shape[:2]
    box_w = x2 - x1
    box_h = y2 - y1
    margin_x = int(box_w * margin_percent / 100)
    margin_y = int(box_h * margin_percent / 100)
    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)
    x2 = min(w, x2 + margin_x)
    y2 = min(h, y2 + margin_y)
    return image[y1:y2, x1:x2]

def assess_quality(image, box):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    grad_diff = abs(sobelx.var() - sobely.var())

    h, w = image.shape[:2]
    hints = []

    if lap_var < 100:
        hints.append(f"blurry [lap_var={lap_var:.1f} < 100]")
    if brightness < 60:
        hints.append(f"too dark [brightness={brightness:.1f} < 60]")
    elif brightness > 200:
        hints.append(f"too bright [brightness={brightness:.1f} > 200]")
    if grad_diff > 400:
        hints.append(f"shaky [shake_score={grad_diff:.1f} > 400]")

    x0, y0, x1, y1 = box
    box_w = x1 - x0
    box_h = y1 - y0
    x_center = (x0 + x1) // 2
    y_center = (y0 + y1) // 2

    if x_center < w * 0.25 or x_center > w * 0.75 or y_center < h * 0.25 or y_center > h * 0.75:
        hints.append("face off-center")
    if box_w < w * 0.25 or box_h < h * 0.25:
        hints.append(f"small face [bbox_w={box_w} < 25%]")
    if x0 < 10 or y0 < 10 or x1 > (w - 10) or y1 > (h - 10):
        hints.append("cropped face")

    return lap_var, brightness, grad_diff, "; ".join(hints) if hints else "good"

def draw_preview(img, face, det_set, score, lap, bright, grad_diff, hint):
    box = face.bbox.astype(int)
    preview = img.copy()

    if score >= 0.90:
        color = (255, 0, 0)
    elif score >= 0.80:
        color = (0, 255, 0)
    elif score >= 0.70:
        color = (0, 255, 255)
    elif score >= 0.60:
        color = (0, 165, 255)
    else:
        color = (0, 0, 255)

    cv2.rectangle(preview, tuple(box[:2]), tuple(box[2:]), color, 2)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = PREVIEW_TEXT_SIZE if ENABLE_CUSTOM_FONT else 0.5
    font_thickness = 2 if PREVIEW_TEXT_BOLD else 1 if ENABLE_CUSTOM_FONT else 1

    lines = [
        f"{det_set} | score: {score:.3f}",
        f"blur: {lap:.0f} | bright: {bright:.0f} | shake: {grad_diff:.0f}",
        f"hint: {hint}"
    ]

    for i, text in enumerate(lines):
        x, y = box[0], box[1] - 10 - (i * 18)
        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        if ENABLE_TEXT_BG:
            overlay = preview.copy()
            cv2.rectangle(overlay, (x, y - th - 4), (x + tw + 2, y + 2), PREVIEW_TEXT_BG_COLOR, -1)
            cv2.addWeighted(overlay, PREVIEW_TEXT_BG_OPACITY, preview, 1 - PREVIEW_TEXT_BG_OPACITY, 0, preview)
        cv2.putText(preview, text, (x, y), font, font_scale, PREVIEW_TEXT_COLOR, font_thickness)

    return preview

any_detector_used = False

with open(log_path, "w") as log:
    log.write("filename,det_set,result,score,bbox_width,bbox_height,x_center,y_center,blur_score,brightness,shake_score,hint\n")

    for idx, fname in enumerate(tqdm(image_files, desc="🔍 Sorting faces")):
        img_path = os.path.join(INPUT_FOLDER, fname)
        image = cv2.imread(img_path)
        if image is None:
            log.write(f"{fname},,bad,0,0,0,0,0,0,0,0,image not loaded\n")
            summary["bad"] = summary.get("bad", 0) + 1
            continue

        detected = False

        for det in DETECTION_SIZES:
            if get_free_gpu_percent() < (100 - MAX_GPU_MEMORY_PERCENT):
                continue
            any_detector_used = True

            app = detectors[det]
            faces = app.get(image)
            if faces:
                best_face = max(faces, key=lambda f: f.det_score)
                score = best_face.det_score
                box = best_face.bbox.astype(int)
                width = box[2] - box[0]
                height = box[3] - box[1]
                x_center = (box[0] + box[2]) // 2
                y_center = (box[1] + box[3]) // 2
                lap_score, bright_score, grad_diff, hint = assess_quality(image, box)

                out_dir = os.path.join(OUTPUT_BASE, det)
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, fname), image)
                log.write(f"{fname},{det},ok,{score:.4f},{width},{height},{x_center},{y_center},{lap_score:.1f},{bright_score:.1f},{grad_diff:.1f},{hint}\n")
                summary[det] = summary.get(det, 0) + 1

                if ENABLE_PREVIEW_IMAGE:
                    preview_base = image.copy()
                    if ENABLE_PREVIEW_OVERLAY:
                        preview_base = draw_preview(preview_base, best_face, det, score, lap_score, bright_score, grad_diff, hint)
                    preview_crop = crop_with_margin(preview_base, box, PREVIEW_CROP_MARGIN_PERCENT)
                    cv2.imwrite(os.path.join(OUTPUT_BASE, "previews", fname), preview_crop)

                detected = True
                break

        if not detected:
            lap_score, bright_score, grad_diff, hint = assess_quality(image, [0, 0, 0, 0])
            log.write(f"{fname},,bad,0,0,0,0,0,{lap_score:.1f},{bright_score:.1f},{grad_diff:.1f},{hint}\n")
            summary["bad"] = summary.get("bad", 0) + 1

if not any_detector_used:
    print("❌ All detection sizes were skipped due to low GPU memory.")
    with open(log_path, "a") as log:
        log.write("\n# WARNING: All detection sizes skipped due to GPU memory limits.\n")
    exit(1)

nvmlShutdown()

with open(log_path, "a") as log:
    log.write("\n# Summary:\n")
    for k, v in summary.items():
        log.write(f"# {k}: {v}\n")
    log.write("\n# Hint Explanations (with thresholds):\n")
    log.write("# blurry - lap_var < 100. Improve focus or use tripod.\n")
    log.write("# too dark - brightness < 60. Add lighting.\n")
    log.write("# too bright - brightness > 200. Reduce exposure.\n")
    log.write("# shaky - gradient imbalance > 400. Hold camera steady.\n")
    log.write("# small face - bbox < 25% of image size. Move closer.\n")
    log.write("# face off-center - subject not centered.\n")
    log.write("# cropped face - bbox touches edge. Reframe image.\n")
    log.write("# good - image passed all checks.\n")

print("✅ Sorting complete.")
print(f"📁 Output: {OUTPUT_BASE}")











# ** Mehdi _  for weak GPUs - very slow
# import os
# import cv2
# import argparse
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# from insightface.app import FaceAnalysis
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
# import onnxruntime
#
# # === Config ===
# DETECTION_SIZES = ["2048x2048", "1600x1600", "1280x1280", "1024x1024", "896x896", "800x800", "768x768", "704x704", "640x640"]
# MAX_GPU_MEMORY_PERCENT = 95
# OUTPUT_UNDER_MEDIA_FOLDER = True
#
# # === Init ===
# parser = argparse.ArgumentParser()
# parser.add_argument("--input", type=str, required=True, help="Input folder of student images")
# args = parser.parse_args()
# INPUT_FOLDER = args.input
#
# TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# OUTPUT_BASE = os.path.join("media", f"Sorted faces ({TIMESTAMP})") if OUTPUT_UNDER_MEDIA_FOLDER else os.path.join(INPUT_FOLDER, f"Sorted faces ({TIMESTAMP})")
# os.makedirs(OUTPUT_BASE, exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "bad"), exist_ok=True)
#
# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)
# def get_free_gpu_percent():
#     mem = nvmlDeviceGetMemoryInfo(handle)
#     return (mem.free / mem.total) * 100
#
# image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
# log_path = os.path.join(OUTPUT_BASE, "sorted_face_log.csv")
# summary = {}
#
# def assess_quality(image, box):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     brightness = np.mean(gray)
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
#     grad_diff = abs(sobelx.var() - sobely.var())
#     return lap_var, brightness, grad_diff
#
# any_detector_used = False
#
# with open(log_path, "w") as log:
#     log.write("filename,det_set,result,score,hint\n")
#
#     for idx, fname in enumerate(tqdm(image_files, desc="🔍 Sorting faces")):
#         img_path = os.path.join(INPUT_FOLDER, fname)
#         image = cv2.imread(img_path)
#         if image is None:
#             log.write(f"{fname},,bad,0,image not loaded\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#             continue
#
#         detected = False
#
#         for det in DETECTION_SIZES:
#             if get_free_gpu_percent() < (100 - MAX_GPU_MEMORY_PERCENT):
#                 continue
#             any_detector_used = True
#             try:
#                 app = FaceAnalysis(name="buffalo_l")
#                 app.prepare(ctx_id=0, det_size=tuple(map(int, det.split("x"))))
#                 faces = app.get(image)
#                 if faces:
#                     best_face = max(faces, key=lambda f: f.det_score)
#                     score = best_face.det_score
#                     hint = "good"
#                     out_dir = os.path.join(OUTPUT_BASE, det)
#                     os.makedirs(out_dir, exist_ok=True)
#                     cv2.imwrite(os.path.join(out_dir, fname), image)
#                     log.write(f"{fname},{det},ok,{score:.4f},{hint}\n")
#                     summary[det] = summary.get(det, 0) + 1
#                     detected = True
#                     break
#             except onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException as e:
#                 log.write(f"{fname},{det},bad,0,ONNX failed: {str(e).splitlines()[0]}\n")
#                 continue
#             except Exception as ex:
#                 log.write(f"{fname},{det},bad,0,Exception: {str(ex).splitlines()[0]}\n")
#                 continue
#
#         if not detected:
#             log.write(f"{fname},,bad,0,No face detected or all detectors failed\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#
# if not any_detector_used:
#     print("❌ All detection sizes were skipped due to low GPU memory.")
#     with open(log_path, "a") as log:
#         log.write("\n# WARNING: All detection sizes skipped due to GPU memory limits.\n")
#     exit(1)
#
# nvmlShutdown()
#
# with open(log_path, "a") as log:
#     log.write("\n# Summary:\n")
#     for k, v in summary.items():
#         log.write(f"# {k}: {v}\n")

















# # ** Mehdi _ Very very very Good
# import os
# import cv2
# import argparse
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# from insightface.app import FaceAnalysis
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
#
# # --- Configuration ---
# # DETECTION_SIZES = ["2048x2048", "1600x1600", "1280x1280", "1024x1024", "896x896", "800x800", "768x768", "704x704", "640x640"]
# DETECTION_SIZES = ["1280x1280", "1024x1024",  "800x800", "640x640"]
# # === Preview Text Rendering Settings ===
# PREVIEW_TEXT_COLOR = (255, 255, 255)       # Font color (BGR)
# PREVIEW_TEXT_SIZE = 0.5                    # Font scale (default: 0.5)
# PREVIEW_TEXT_BOLD = False                   # Toggle bold text (thickness 2)
# PREVIEW_TEXT_BG_COLOR = (0, 0, 0)          # Text background color (BGR)
# PREVIEW_TEXT_BG_OPACITY = 0.6              # Background opacity (0.0 to 1.0)
# ENABLE_TEXT_BG = True                      # Enable/disable text background
# ENABLE_CUSTOM_FONT = True                  # Enable font color/size/thickness
#
# ENABLE_PREVIEW_IMAGE = True
# ENABLE_PREVIEW_OVERLAY = True
# PREVIEW_CROP_MARGIN_PERCENT = 30
# ENABLE_TERMINAL_LOGS = True
# OUTPUT_UNDER_MEDIA_FOLDER = True
# MAX_GPU_MEMORY_PERCENT = 80
# MEMORY_CHECK_INTERVAL = 10
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--input", type=str, required=True, help="Input folder of student images")
# args = parser.parse_args()
#
# INPUT_FOLDER = args.input
# TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# OUTPUT_BASE = os.path.join("media", f"Sorted faces ({TIMESTAMP})") if OUTPUT_UNDER_MEDIA_FOLDER else os.path.join(INPUT_FOLDER, f"Sorted faces ({TIMESTAMP})")
# os.makedirs(OUTPUT_BASE, exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "previews"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "bad"), exist_ok=True)
#
# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)
# def get_free_gpu_percent():
#     mem = nvmlDeviceGetMemoryInfo(handle)
#     return (mem.free / mem.total) * 100
#
# # Track if at least one detector ran
# any_detector_used = False
#
# detectors = {}
# for size in DETECTION_SIZES:
#     w, h = map(int, size.split("x"))
#     app = FaceAnalysis(name="buffalo_l")
#     app.prepare(ctx_id=0, det_size=(w, h))
#     detectors[size] = app
#
# image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
# log_path = os.path.join(OUTPUT_BASE, "sorted_face_log.csv")
# summary = {}
#
# def crop_with_margin(image, box, margin_percent):
#     if margin_percent is None:
#         return image  # ✅ No cropping if None
#     x1, y1, x2, y2 = box
#     h, w = image.shape[:2]
#     box_w = x2 - x1
#     box_h = y2 - y1
#     margin_x = int(box_w * margin_percent / 100)
#     margin_y = int(box_h * margin_percent / 100)
#     x1 = max(0, x1 - margin_x)
#     y1 = max(0, y1 - margin_y)
#     x2 = min(w, x2 + margin_x)
#     y2 = min(h, y2 + margin_y)
#     return image[y1:y2, x1:x2]
#
#
# def assess_quality(image, box):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     brightness = np.mean(gray)
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
#     var_x = sobelx.var()
#     var_y = sobely.var()
#     grad_diff = abs(var_x - var_y)
#     h, w = image.shape[:2]
#     hints = []
#
#     if lap_var < 100:
#         hints.append(f"blurry [lap_var={lap_var:.1f} < 100]")
#     if brightness < 60:
#         hints.append(f"too dark [brightness={brightness:.1f} < 60]")
#     elif brightness > 200:
#         hints.append(f"too bright [brightness={brightness:.1f} > 200]")
#     if grad_diff > 400:
#         hints.append(f"shaky [shake_score={grad_diff:.1f} > 400]")
#
#     x0, y0, x1, y1 = box
#     box_w = x1 - x0
#     box_h = y1 - y0
#     x_center = (x0 + x1) // 2
#     y_center = (y0 + y1) // 2
#
#     if x_center < w * 0.25 or x_center > w * 0.75 or y_center < h * 0.25 or y_center > h * 0.75:
#         hints.append("face off-center")
#     if box_w < w * 0.25 or box_h < h * 0.25:
#         hints.append(f"small face [bbox_w={box_w} < 25%]")
#     if x0 < 10 or y0 < 10 or x1 > (w - 10) or y1 > (h - 10):
#         hints.append("cropped face")
#
#     return lap_var, brightness, grad_diff, "; ".join(hints) if hints else "good"
#
# def draw_preview(img, face, det_set, score, lap, bright, grad_diff, hint):
#     box = face.bbox.astype(int)
#     preview = img.copy()
#
#     # Color code the box by score
#     if score >= 0.90:
#         color = (255, 0, 0)       # Blue
#     elif score >= 0.80:
#         color = (0, 255, 0)       # Green
#     elif score >= 0.70:
#         color = (0, 255, 255)     # Yellow
#     elif score >= 0.60:
#         color = (0, 165, 255)     # Orange
#     else:
#         color = (0, 0, 255)       # Red
#
#     cv2.rectangle(preview, tuple(box[:2]), tuple(box[2:]), color, 2)
#
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     font_scale = PREVIEW_TEXT_SIZE if ENABLE_CUSTOM_FONT else 0.5
#     font_thickness = 2 if PREVIEW_TEXT_BOLD else 1 if ENABLE_CUSTOM_FONT else 1
#     text_color = PREVIEW_TEXT_COLOR
#
#     lines = [
#         f"{det_set} | score: {score:.3f}",
#         f"blur: {lap:.0f} | bright: {bright:.0f} | shake: {grad_diff:.0f}",
#         f"hint: {hint}"
#     ]
#
#     for i, text in enumerate(lines):
#         x, y = box[0], box[1] - 10 - (i * 18)
#         (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
#         if ENABLE_TEXT_BG:
#             overlay = preview.copy()
#             cv2.rectangle(overlay, (x, y - th - 4), (x + tw + 2, y + 2), PREVIEW_TEXT_BG_COLOR, -1)
#             cv2.addWeighted(overlay, PREVIEW_TEXT_BG_OPACITY, preview, 1 - PREVIEW_TEXT_BG_OPACITY, 0, preview)
#         cv2.putText(preview, text, (x, y), font, font_scale, text_color, font_thickness)
#
#     return preview
#
#
#
# with open(log_path, "w") as log:
#     log.write("filename,det_set,result,score,bbox_width,bbox_height,x_center,y_center,blur_score,brightness,shake_score,hint\n")
#
#     for idx, fname in enumerate(tqdm(image_files, desc="🔍 Sorting faces")):
#         img_path = os.path.join(INPUT_FOLDER, fname)
#         image = cv2.imread(img_path)
#         if image is None:
#             log.write(f"{fname},,bad,0,0,0,0,0,0,0,0,image not loaded\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#             continue
#
#         detected = False
#
#         if idx % MEMORY_CHECK_INTERVAL == 0:
#             if get_free_gpu_percent() < (100 - MAX_GPU_MEMORY_PERCENT) and ENABLE_TERMINAL_LOGS:
#                 print(f"[INFO] GPU memory low, skipping larger det_sets...")
#
#         for det in DETECTION_SIZES:
#             if get_free_gpu_percent() < (100 - MAX_GPU_MEMORY_PERCENT):
#                 continue
#
#             app = detectors[det]
#             faces = app.get(image)
#             if faces:
#                 best_face = max(faces, key=lambda f: f.det_score)
#                 score = best_face.det_score
#                 box = best_face.bbox.astype(int)
#                 width = box[2] - box[0]
#                 height = box[3] - box[1]
#                 x_center = (box[0] + box[2]) // 2
#                 y_center = (box[1] + box[3]) // 2
#                 lap_score, bright_score, grad_diff, hint = assess_quality(image, box)
#
#                 out_dir = os.path.join(OUTPUT_BASE, det)
#                 os.makedirs(out_dir, exist_ok=True)
#                 cv2.imwrite(os.path.join(out_dir, fname), image)
#                 log.write(f"{fname},{det},ok,{score:.4f},{width},{height},{x_center},{y_center},{lap_score:.1f},{bright_score:.1f},{grad_diff:.1f},{hint}\n")
#                 summary[det] = summary.get(det, 0) + 1
#
#                 if ENABLE_PREVIEW_IMAGE:
#                     # ✅ Always draw on the original image before cropping
#                     preview_base = image.copy()
#                     if ENABLE_PREVIEW_OVERLAY:
#                         preview_base = draw_preview(preview_base, best_face, det, score, lap_score, bright_score,
#                                                     grad_diff, hint)
#                     preview_crop = crop_with_margin(preview_base, box, PREVIEW_CROP_MARGIN_PERCENT)
#                     cv2.imwrite(os.path.join(OUTPUT_BASE, "previews", fname), preview_crop)
#
#                 detected = True
#                 break
#
#         if not detected:
#             lap_score, bright_score, grad_diff, hint = assess_quality(image, [0, 0, 0, 0])
#             log.write(f"{fname},,bad,0,0,0,0,0,{lap_score:.1f},{bright_score:.1f},{grad_diff:.1f},{hint}\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#
# nvmlShutdown()
#
# with open(log_path, "a") as log:
#     log.write("\n# Summary:\n")
#     for k, v in summary.items():
#         log.write(f"# {k}: {v}\n")
#     log.write("\n# Hint Explanations (with thresholds):\n")
#     log.write("# blurry - Image is out of focus [lap_var < 100]. Improve focus or use a tripod.\n")
#     log.write("# too dark - Brightness < 60. Increase ambient lighting.\n")
#     log.write("# too bright - Brightness > 200. Reduce exposure or avoid harsh lights.\n")
#     log.write("# shaky - Motion blur detected [abs(grad_x - grad_y) > 400]. Hold the camera steady.\n")
#     log.write("# small face - Face bounding box width/height < 25% of image size. Move student closer.\n")
#     log.write("# face off-center - Face center outside 25%-75% of image width/height. Center the subject.\n")
#     log.write("# cropped face - Face touches frame edge. Ensure full head and shoulders are in the frame.\n")
#     log.write("# good - Image meets all quality standards.\n")
#
# print("✅ Sorting complete.")
# print(f"📁 Output: {OUTPUT_BASE}")









# # ** Mehdi very goog script _ I want a little more.
# import os
# import cv2
# import argparse
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# from insightface.app import FaceAnalysis
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
#
# # DETECTION_SIZES = ["2048x2048", "1600x1600", "1280x1280", "1024x1024", "896x896", "800x800", "768x768", "704x704", "640x640"]
# DETECTION_SIZES = ["1280x1280", "1024x1024",  "800x800", "640x640"]
# ENABLE_PREVIEW_OVERLAY = True
# ENABLE_TERMINAL_LOGS = True
# OUTPUT_UNDER_MEDIA_FOLDER = True
# MAX_GPU_MEMORY_PERCENT = 80
# MEMORY_CHECK_INTERVAL = 10
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--input", type=str, required=True, help="Input folder of student images")
# args = parser.parse_args()
#
# INPUT_FOLDER = args.input
# TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# OUTPUT_BASE = os.path.join("media", f"Sorted faces ({TIMESTAMP})") if OUTPUT_UNDER_MEDIA_FOLDER else os.path.join(INPUT_FOLDER, f"Sorted faces ({TIMESTAMP})")
# os.makedirs(OUTPUT_BASE, exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "previews"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "bad"), exist_ok=True)
#
# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)
# def get_free_gpu_percent():
#     mem = nvmlDeviceGetMemoryInfo(handle)
#     return (mem.free / mem.total) * 100
#
# detectors = {}
# for size in DETECTION_SIZES:
#     w, h = map(int, size.split("x"))
#     app = FaceAnalysis(name="buffalo_l")
#     app.prepare(ctx_id=0, det_size=(w, h))
#     detectors[size] = app
#
# image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
# log_path = os.path.join(OUTPUT_BASE, "sorted_face_log.csv")
# summary = {}
#
# def assess_quality(image, box):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     brightness = np.mean(gray)
#
#     # Shaky detection using Sobel gradient imbalance
#     sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
#     sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
#     var_x = sobelx.var()
#     var_y = sobely.var()
#     grad_diff = abs(var_x - var_y)
#
#     h, w = image.shape[:2]
#     hints = []
#
#     if lap_var < 100:
#         hints.append("blurry")
#     if brightness < 60:
#         hints.append("too dark")
#     elif brightness > 200:
#         hints.append("too bright")
#     if grad_diff > 400:
#         hints.append("shaky")
#
#     x0, y0, x1, y1 = box
#     box_w = x1 - x0
#     box_h = y1 - y0
#     x_center = (x0 + x1) // 2
#     y_center = (y0 + y1) // 2
#
#     if x_center < w * 0.25 or x_center > w * 0.75 or y_center < h * 0.25 or y_center > h * 0.75:
#         hints.append("face off-center")
#     if box_w < w * 0.25 or box_h < h * 0.25:
#         hints.append("small face")
#     if x0 < 10 or y0 < 10 or x1 > (w - 10) or y1 > (h - 10):
#         hints.append("cropped face")
#
#     return lap_var, brightness, grad_diff, "; ".join(hints) if hints else "good"
#
# def draw_preview(img, face, det_set, score, lap, bright, grad_diff, hint):
#     preview = img.copy()
#     box = face.bbox.astype(int)
#     cv2.rectangle(preview, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
#     lines = [
#         f"{det_set} | score: {score:.3f}",
#         f"blur: {lap:.0f} | bright: {bright:.0f} | shake: {grad_diff:.0f}",
#         f"hint: {hint}"
#     ]
#     for i, text in enumerate(lines):
#         y = box[1] - 10 - (i * 18)
#         cv2.putText(preview, text, (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     return preview
#
# with open(log_path, "w") as log:
#     log.write("filename,det_set,result,score,bbox_width,bbox_height,x_center,y_center,blur_score,brightness,shake_score,hint\n")
#
#     for idx, fname in enumerate(tqdm(image_files, desc="🔍 Sorting faces")):
#         img_path = os.path.join(INPUT_FOLDER, fname)
#         image = cv2.imread(img_path)
#         if image is None:
#             log.write(f"{fname},,bad,0,0,0,0,0,0,0,0,image not loaded\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#             print(f"[ERROR] Failed to load image: {img_path}")
#             continue
#
#         detected = False
#
#         if idx % MEMORY_CHECK_INTERVAL == 0:
#             free_mem = get_free_gpu_percent()
#             if free_mem < (100 - MAX_GPU_MEMORY_PERCENT) and ENABLE_TERMINAL_LOGS:
#                 print(f"[INFO] GPU memory low ({int(free_mem)}%), may skip larger det_sets")
#
#         for det in DETECTION_SIZES:
#             if get_free_gpu_percent() < (100 - MAX_GPU_MEMORY_PERCENT):
#                 continue
#
#             app = detectors[det]
#             faces = app.get(image)
#             if faces:
#                 best_face = max(faces, key=lambda f: f.det_score)
#                 score = best_face.det_score
#                 box = best_face.bbox.astype(int)
#                 width = box[2] - box[0]
#                 height = box[3] - box[1]
#                 x_center = (box[0] + box[2]) // 2
#                 y_center = (box[1] + box[3]) // 2
#                 lap_score, bright_score, grad_diff, hint = assess_quality(image, box)
#
#                 out_dir = os.path.join(OUTPUT_BASE, det)
#                 os.makedirs(out_dir, exist_ok=True)
#                 cv2.imwrite(os.path.join(out_dir, fname), image)
#                 log.write(f"{fname},{det},ok,{score:.4f},{width},{height},{x_center},{y_center},{lap_score:.1f},{bright_score:.1f},{grad_diff:.1f},{hint}\n")
#                 summary[det] = summary.get(det, 0) + 1
#
#                 if ENABLE_PREVIEW_OVERLAY:
#                     preview_img = draw_preview(image, best_face, det, score, lap_score, bright_score, grad_diff, hint)
#                     cv2.imwrite(os.path.join(OUTPUT_BASE, "previews", fname), preview_img)
#                 detected = True
#                 break
#
#         if not detected:
#             lap_score, bright_score, grad_diff, hint = assess_quality(image, [0, 0, 0, 0])
#             log.write(f"{fname},,bad,0,0,0,0,0,{lap_score:.1f},{bright_score:.1f},{grad_diff:.1f},{hint}\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#
# nvmlShutdown()
#
# with open(log_path, "a") as log:
#     log.write("\n# Summary:\n")
#     for k, v in summary.items():
#         log.write(f"# {k}: {v}\n")
#
#     log.write("\n# Hint Explanations (with thresholds):\n")
#     log.write("# blurry - Image is out of focus [lap_var < 100]. Improve focus or use a tripod.\n")
#     log.write("# too dark - Brightness < 60. Increase ambient lighting.\n")
#     log.write("# too bright - Brightness > 200. Reduce exposure or avoid harsh lights.\n")
#     log.write("# shaky - Motion blur detected [abs(grad_x - grad_y) > 400]. Hold the camera steady.\n")
#     log.write("# small face - Face bounding box width/height < 25% of image size. Move student closer.\n")
#     log.write("# face off-center - Face center outside 25%-75% of image width/height. Center the subject.\n")
#     log.write("# cropped face - Face touches frame edge. Ensure full head and shoulders are in the frame.\n")
#     log.write("# good - Image meets all quality standards.\n")
#
# print("✅ Sorting complete.")
# print(f"📁 Output: {OUTPUT_BASE}")












# import os
# import cv2
# import argparse
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# from insightface.app import FaceAnalysis
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
#
# # DETECTION_SIZES = ["2048x2048", "1600x1600", "1280x1280", "1024x1024", "896x896", "800x800", "768x768", "704x704", "640x640"]
# DETECTION_SIZES = ["1280x1280", "1024x1024",  "800x800", "640x640"]
# ENABLE_PREVIEW_OVERLAY = True
# ENABLE_TERMINAL_LOGS = True
# OUTPUT_UNDER_MEDIA_FOLDER = True
# MAX_GPU_MEMORY_PERCENT = 80
# MEMORY_CHECK_INTERVAL = 10
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--input", type=str, required=True, help="Input folder of student images")
# args = parser.parse_args()
#
# INPUT_FOLDER = args.input
# TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# OUTPUT_BASE = os.path.join("media", f"Sorted faces ({TIMESTAMP})") if OUTPUT_UNDER_MEDIA_FOLDER else os.path.join(INPUT_FOLDER, f"Sorted faces ({TIMESTAMP})")
# os.makedirs(OUTPUT_BASE, exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "previews"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "bad"), exist_ok=True)
#
# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)
# def get_free_gpu_percent():
#     mem = nvmlDeviceGetMemoryInfo(handle)
#     return (mem.free / mem.total) * 100
#
# detectors = {}
# for size in DETECTION_SIZES:
#     w, h = map(int, size.split("x"))
#     app = FaceAnalysis(name="buffalo_l")
#     app.prepare(ctx_id=0, det_size=(w, h))
#     detectors[size] = app
#
# image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
# log_path = os.path.join(OUTPUT_BASE, "sorted_face_log.csv")
# summary = {}
#
# def assess_quality(image, box):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     brightness = np.mean(gray)
#     h, w = image.shape[:2]
#     hints = []
#
#     if lap_var < 100:
#         hints.append("blurry")
#     if brightness < 60:
#         hints.append("too dark")
#     elif brightness > 200:
#         hints.append("too bright")
#
#     x0, y0, x1, y1 = box
#     box_w = x1 - x0
#     box_h = y1 - y0
#     x_center = (x0 + x1) // 2
#     y_center = (y0 + y1) // 2
#
#     if x_center < w * 0.25 or x_center > w * 0.75 or y_center < h * 0.25 or y_center > h * 0.75:
#         hints.append("face off-center")
#     if box_w < w * 0.25 or box_h < h * 0.25:
#         hints.append("small face")
#     if x0 < 10 or y0 < 10 or x1 > (w - 10) or y1 > (h - 10):
#         hints.append("cropped face")
#
#     return lap_var, brightness, "; ".join(hints) if hints else "good"
#
# def draw_preview(img, face, det_set, score, lap, bright, hint):
#     preview = img.copy()
#     box = face.bbox.astype(int)
#     cv2.rectangle(preview, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
#     lines = [
#         f"{det_set} | score: {score:.3f}",
#         f"blur: {lap:.0f} | bright: {bright:.0f}",
#         f"hint: {hint}"
#     ]
#     for i, text in enumerate(lines):
#         y = box[1] - 10 - (i * 18)
#         cv2.putText(preview, text, (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     return preview
#
# with open(log_path, "w") as log:
#     log.write("filename,det_set,result,score,bbox_width,bbox_height,x_center,y_center,blur_score,brightness,hint\n")
#
#     for idx, fname in enumerate(tqdm(image_files, desc="🔍 Sorting faces")):
#         img_path = os.path.join(INPUT_FOLDER, fname)
#         image = cv2.imread(img_path)
#         if image is None:
#             log.write(f"{fname},,bad,0,0,0,0,0,0,0,image not loaded\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#             print(f"[ERROR] Failed to load image: {img_path}")
#             continue
#
#         detected = False
#
#         if idx % MEMORY_CHECK_INTERVAL == 0:
#             free_mem = get_free_gpu_percent()
#             if free_mem < (100 - MAX_GPU_MEMORY_PERCENT) and ENABLE_TERMINAL_LOGS:
#                 print(f"[INFO] GPU memory low ({int(free_mem)}%), may skip larger det_sets")
#
#         for det in DETECTION_SIZES:
#             if get_free_gpu_percent() < (100 - MAX_GPU_MEMORY_PERCENT):
#                 continue
#
#             app = detectors[det]
#             faces = app.get(image)
#             if faces:
#                 best_face = max(faces, key=lambda f: f.det_score)
#                 score = best_face.det_score
#                 box = best_face.bbox.astype(int)
#                 width = box[2] - box[0]
#                 height = box[3] - box[1]
#                 x_center = (box[0] + box[2]) // 2
#                 y_center = (box[1] + box[3]) // 2
#                 lap_score, bright_score, hint = assess_quality(image, box)
#
#                 out_dir = os.path.join(OUTPUT_BASE, det)
#                 os.makedirs(out_dir, exist_ok=True)
#                 cv2.imwrite(os.path.join(out_dir, fname), image)
#                 log.write(f"{fname},{det},ok,{score:.4f},{width},{height},{x_center},{y_center},{lap_score:.1f},{bright_score:.1f},{hint}\n")
#                 summary[det] = summary.get(det, 0) + 1
#
#                 if ENABLE_PREVIEW_OVERLAY:
#                     preview_img = draw_preview(image, best_face, det, score, lap_score, bright_score, hint)
#                     cv2.imwrite(os.path.join(OUTPUT_BASE, "previews", fname), preview_img)
#                 detected = True
#                 break
#
#         if not detected:
#             lap_score, bright_score, hint = assess_quality(image, [0, 0, 0, 0])
#             log.write(f"{fname},,bad,0,0,0,0,0,{lap_score:.1f},{bright_score:.1f},{hint}\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#
# nvmlShutdown()
#
# with open(log_path, "a") as log:
#     log.write("\n# Summary:\n")
#     for k, v in summary.items():
#         log.write(f"# {k}: {v}\n")
#     log.write("\n# Hint Explanations:\n")
#     log.write("# blurry - Laplacian variance < 100 → image is out of focus. Retake with stable hands or tripod.\n")
#     log.write("# too dark - Brightness < 60 → image is underexposed. Use better lighting.\n")
#     log.write("# too bright - Brightness > 200 → image is overexposed. Avoid strong direct light.\n")
#     log.write("# face off-center - Face not centered → position student in middle of frame.\n")
#     log.write("# small face - Face bbox < 25% of image → move student closer.\n")
#     log.write("# cropped face - Face touches edge of image → ensure full head is inside frame.\n")
#     log.write("# good - Image quality meets standards for recognition.\n")
#
# print("✅ Sorting complete.")
# print(f"📁 Output: {OUTPUT_BASE}")











# import os
# import cv2
# import argparse
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# from insightface.app import FaceAnalysis
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
#
# # DETECTION_SIZES = ["2048x2048", "1600x1600", "1280x1280", "1024x1024", "896x896", "800x800", "768x768", "704x704", "640x640"]
# DETECTION_SIZES = ["1280x1280", "1024x1024",  "800x800", "640x640"]
# ENABLE_PREVIEW_OVERLAY = True
# ENABLE_TERMINAL_LOGS = True
# OUTPUT_UNDER_MEDIA_FOLDER = True
# MAX_GPU_MEMORY_PERCENT = 80
# MEMORY_CHECK_INTERVAL = 10
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--input", type=str, required=True, help="Input folder of student images")
# args = parser.parse_args()
#
# INPUT_FOLDER = args.input
# TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# OUTPUT_BASE = os.path.join("media", f"Sorted faces ({TIMESTAMP})") if OUTPUT_UNDER_MEDIA_FOLDER else os.path.join(INPUT_FOLDER, f"Sorted faces ({TIMESTAMP})")
# os.makedirs(OUTPUT_BASE, exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "previews"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "bad"), exist_ok=True)
#
# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)
# def get_free_gpu_percent():
#     mem = nvmlDeviceGetMemoryInfo(handle)
#     return (mem.free / mem.total) * 100
#
# detectors = {}
# for size in DETECTION_SIZES:
#     w, h = map(int, size.split("x"))
#     app = FaceAnalysis(name="buffalo_l")
#     app.prepare(ctx_id=0, det_size=(w, h))
#     detectors[size] = app
#
# image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
# log_path = os.path.join(OUTPUT_BASE, "sorted_face_log.csv")
# summary = {}
#
# def assess_quality(image, box):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     brightness = np.mean(gray)
#     hints = []
#     if laplacian_var < 100:
#         hints.append("blurry")
#     if brightness < 60:
#         hints.append("too dark")
#     elif brightness > 200:
#         hints.append("too bright")
#     return laplacian_var, brightness, "; ".join(hints) if hints else "good"
#
# def draw_preview(img, face, det_set, score, lap, bright, hint):
#     preview = img.copy()
#     box = face.bbox.astype(int)
#     cv2.rectangle(preview, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
#     lines = [
#         f"{det_set} | score: {score:.3f}",
#         f"blur: {lap:.0f} | bright: {bright:.0f}",
#         f"hint: {hint}"
#     ]
#     for i, text in enumerate(lines):
#         y = box[1] - 10 - (i * 18)
#         cv2.putText(preview, text, (box[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
#     return preview
#
# with open(log_path, "w") as log:
#     log.write("filename,det_set,result,score,bbox_width,bbox_height,x_center,y_center,blur_score,brightness,hint\n")
#
#     for idx, fname in enumerate(tqdm(image_files, desc="🔍 Sorting faces")):
#         img_path = os.path.join(INPUT_FOLDER, fname)
#         image = cv2.imread(img_path)
#         if image is None:
#             log.write(f"{fname},,bad,0,0,0,0,0,0,0,image not loaded\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#             print(f"[ERROR] Failed to load image: {img_path}")
#             continue
#
#         detected = False
#
#         if idx % MEMORY_CHECK_INTERVAL == 0:
#             free_mem = get_free_gpu_percent()
#             if free_mem < (100 - MAX_GPU_MEMORY_PERCENT) and ENABLE_TERMINAL_LOGS:
#                 print(f"[INFO] GPU memory low ({int(free_mem)}%), may skip larger det_sets")
#
#         for det in DETECTION_SIZES:
#             if get_free_gpu_percent() < (100 - MAX_GPU_MEMORY_PERCENT):
#                 continue
#
#             app = detectors[det]
#             faces = app.get(image)
#             if faces:
#                 best_face = max(faces, key=lambda f: f.det_score)
#                 score = best_face.det_score
#                 box = best_face.bbox.astype(int)
#                 width = box[2] - box[0]
#                 height = box[3] - box[1]
#                 x_center = (box[0] + box[2]) // 2
#                 y_center = (box[1] + box[3]) // 2
#                 lap_score, bright_score, hint = assess_quality(image, box)
#
#                 out_dir = os.path.join(OUTPUT_BASE, det)
#                 os.makedirs(out_dir, exist_ok=True)
#                 cv2.imwrite(os.path.join(out_dir, fname), image)
#                 log.write(f"{fname},{det},ok,{score:.4f},{width},{height},{x_center},{y_center},{lap_score:.1f},{bright_score:.1f},{hint}\n")
#                 summary[det] = summary.get(det, 0) + 1
#
#                 if ENABLE_PREVIEW_OVERLAY:
#                     preview_img = draw_preview(image, best_face, det, score, lap_score, bright_score, hint)
#                     cv2.imwrite(os.path.join(OUTPUT_BASE, "previews", fname), preview_img)
#                 detected = True
#                 break
#
#         if not detected:
#             lap_score, bright_score, hint = assess_quality(image, [0, 0, 0, 0])
#             log.write(f"{fname},,bad,0,0,0,0,0,{lap_score:.1f},{bright_score:.1f},{hint}\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#
# nvmlShutdown()
#
# with open(log_path, "a") as log:
#     log.write("\n# Summary:\n")
#     for k, v in summary.items():
#         log.write(f"# {k}: {v}\n")
#
# print("✅ Sorting complete.")
# print(f"📁 Output: {OUTPUT_BASE}")

















#  ** Mehdi _ Good but I want better
# import os
# import cv2
# import argparse
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# from insightface.app import FaceAnalysis
# from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown
#
# # --- Configuration ---
# # DETECTION_SIZES = ["2048x2048", "1600x1600", "1280x1280", "1024x1024", "896x896", "800x800", "768x768", "704x704", "640x640"]
# DETECTION_SIZES = ["1280x1280", "1024x1024",  "800x800", "640x640"]
# ENABLE_PREVIEW_OVERLAY = True
# ENABLE_TERMINAL_LOGS = True
# OUTPUT_UNDER_MEDIA_FOLDER = True
# MAX_GPU_MEMORY_PERCENT = 80
# MEMORY_CHECK_INTERVAL = 10
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--input", type=str, required=True, help="Input folder of student images")
# args = parser.parse_args()
#
# INPUT_FOLDER = args.input
# TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# OUTPUT_BASE = os.path.join("media", f"Sorted faces ({TIMESTAMP})") if OUTPUT_UNDER_MEDIA_FOLDER else os.path.join(INPUT_FOLDER, f"Sorted faces ({TIMESTAMP})")
# os.makedirs(OUTPUT_BASE, exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "previews"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "bad"), exist_ok=True)
#
# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)
# def get_free_gpu_percent():
#     mem = nvmlDeviceGetMemoryInfo(handle)
#     return (mem.free / mem.total) * 100
#
# detectors = {}
# for size in DETECTION_SIZES:
#     w, h = map(int, size.split("x"))
#     app = FaceAnalysis(name="buffalo_l")
#     app.prepare(ctx_id=0, det_size=(w, h))
#     detectors[size] = app
#
# image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
# log_path = os.path.join(OUTPUT_BASE, "sorted_face_log.csv")
# summary = {}
#
# def draw_preview(img, face, det_set, score):
#     preview = img.copy()
#     box = face.bbox.astype(int)
#     cv2.rectangle(preview, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
#     text = f"{det_set} | score: {score:.3f}"
#     cv2.putText(preview, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#     return preview
#
# with open(log_path, "w") as log:
#     log.write("filename,det_set,result,score,bbox_width,bbox_height,x_center,y_center\n")
#
#     for idx, fname in enumerate(tqdm(image_files, desc="🔍 Sorting faces")):
#         img_path = os.path.join(INPUT_FOLDER, fname)
#         image = cv2.imread(img_path)
#         if image is None:
#             log.write(f"{fname},,bad,0,0,0,0,0\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#             print(f"[ERROR] Failed to load image: {img_path}")
#             continue
#
#         detected = False
#
#         if idx % MEMORY_CHECK_INTERVAL == 0:
#             free_mem = get_free_gpu_percent()
#             if free_mem < (100 - MAX_GPU_MEMORY_PERCENT) and ENABLE_TERMINAL_LOGS:
#                 print(f"[INFO] GPU memory low ({int(free_mem)}%), may skip larger det_sets")
#
#         for det in DETECTION_SIZES:
#             if get_free_gpu_percent() < (100 - MAX_GPU_MEMORY_PERCENT):
#                 continue
#
#             app = detectors[det]
#             faces = app.get(image)
#             if faces:
#                 best_face = max(faces, key=lambda f: f.det_score)
#                 score = best_face.det_score
#                 box = best_face.bbox.astype(int)
#                 width = box[2] - box[0]
#                 height = box[3] - box[1]
#                 x_center = (box[0] + box[2]) // 2
#                 y_center = (box[1] + box[3]) // 2
#
#                 out_dir = os.path.join(OUTPUT_BASE, det)
#                 os.makedirs(out_dir, exist_ok=True)
#                 cv2.imwrite(os.path.join(out_dir, fname), image)
#                 log.write(f"{fname},{det},ok,{score:.4f},{width},{height},{x_center},{y_center}\n")
#                 summary[det] = summary.get(det, 0) + 1
#
#                 if ENABLE_PREVIEW_OVERLAY:
#                     preview_img = draw_preview(image, best_face, det, score)
#                     cv2.imwrite(os.path.join(OUTPUT_BASE, "previews", fname), preview_img)
#                 detected = True
#                 break
#
#         if not detected:
#             log.write(f"{fname},,bad,0,0,0,0,0\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#
# nvmlShutdown()
#
# with open(log_path, "a") as log:
#     log.write("\n# Summary:\n")
#     for k, v in summary.items():
#         log.write(f"# {k}: {v}\n")
#
# print("✅ Sorting complete.")
# print(f"📁 Output: {OUTPUT_BASE}")









# # ** Mehdi _ Very good script
# import os
# import cv2
# import argparse
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# from insightface.app import FaceAnalysis
# from pynvml import (
#     nvmlInit, nvmlDeviceGetHandleByIndex,
#     nvmlDeviceGetMemoryInfo, nvmlShutdown
# )
#
# # --- Configuration ---
# DETECTION_SIZES = ["2048x2048", "1024x1024", "800x800", "640x640"]
# ENABLE_PREVIEW_OVERLAY = True
# ENABLE_TERMINAL_LOGS = True
# OUTPUT_UNDER_MEDIA_FOLDER = True
# FACE_MARGIN_RATIO_CROP = 0.2
# FACE_MARGIN_RATIO_PREVIEW = 0.4
# PERCENT_KEEP_FACE_BOX_PREVIEW = 1.0
# MAX_GPU_MEMORY_PERCENT = 80
# MEMORY_CHECK_INTERVAL = 10  # check every N images
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--input", type=str, required=True, help="Input folder of student images")
# args = parser.parse_args()
#
# INPUT_FOLDER = args.input
# TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# OUTPUT_BASE = os.path.join("media", f"Sorted faces ({TIMESTAMP})") if OUTPUT_UNDER_MEDIA_FOLDER else os.path.join(INPUT_FOLDER, f"Sorted faces ({TIMESTAMP})")
# os.makedirs(OUTPUT_BASE, exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "previews"), exist_ok=True)
# os.makedirs(os.path.join(OUTPUT_BASE, "bad"), exist_ok=True)
#
# # Init GPU monitoring
# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)
# def get_free_gpu_percent():
#     mem = nvmlDeviceGetMemoryInfo(handle)
#     return (mem.free / mem.total) * 100
#
# # Init one FaceAnalysis per det_set
# detectors = {}
# for size in DETECTION_SIZES:
#     w, h = map(int, size.split("x"))
#     app = FaceAnalysis(name="buffalo_l")
#     app.prepare(ctx_id=0, det_size=(w, h))
#     detectors[size] = app
#
# # Sorted file list
# image_files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
# log_path = os.path.join(OUTPUT_BASE, "sorted_face_log.csv")
# summary = {}
#
# with open(log_path, "w") as log:
#     log.write("filename,det_set,result\n")
#
#     for idx, fname in enumerate(tqdm(image_files, desc="🔍 Sorting faces")):
#         img_path = os.path.join(INPUT_FOLDER, fname)
#         image = cv2.imread(img_path)
#         if image is None:
#             log.write(f"{fname},,bad\\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#             print(f"[ERROR] Failed to load image: {img_path}")
#             continue
#
#         detected = False
#
#         if idx % MEMORY_CHECK_INTERVAL == 0:
#             free_mem = get_free_gpu_percent()
#             if free_mem < (100 - MAX_GPU_MEMORY_PERCENT) and ENABLE_TERMINAL_LOGS:
#                 print(f"[INFO] GPU memory low ({int(free_mem)}%), may skip larger det_sets")
#
#         for det in DETECTION_SIZES:
#             if get_free_gpu_percent() < (100 - MAX_GPU_MEMORY_PERCENT):
#                 continue
#
#             app = detectors[det]
#             faces = app.get(image)
#             if faces:
#                 out_dir = os.path.join(OUTPUT_BASE, det)
#                 os.makedirs(out_dir, exist_ok=True)
#                 cv2.imwrite(os.path.join(out_dir, fname), image)
#                 log.write(f"{fname},{det},ok\n")
#                 summary[det] = summary.get(det, 0) + 1
#                 detected = True
#
#                 if ENABLE_PREVIEW_OVERLAY:
#                     preview = image.copy()
#                     for face in faces:
#                         box = face.bbox.astype(int)
#                         cv2.rectangle(preview, tuple(box[:2]), tuple(box[2:]), (0,255,0), 2)
#                         cv2.putText(preview, det, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
#                     preview_path = os.path.join(OUTPUT_BASE, "previews", fname)
#                     cv2.imwrite(preview_path, preview)
#
#                 break
#
#         if not detected:
#             cv2.imwrite(os.path.join(OUTPUT_BASE, "bad", fname), image)
#             log.write(f"{fname},,bad\n")
#             summary["bad"] = summary.get("bad", 0) + 1
#
# nvmlShutdown()
#
# with open(log_path, "a") as log:
#     log.write("\n# Summary:\n")
#     for k, v in summary.items():
#         log.write(f"# {k}: {v}\n")
#
# print("✅ Sorting complete.")
# print(f"📁 Output: {OUTPUT_BASE}")











# #!/usr/bin/env python3 (Good but only if there is no pressure on GPU)
# import os
# import sys
# import cv2
# import csv
# import shutil
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# import argparse
# from concurrent.futures import ThreadPoolExecutor
# from threading import Lock
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # ======================= CONFIGURATION ===========================
# AUTO_ROUND_DETSET = False
# MAX_PARALLEL_IMAGES = 1
# SAVE_DIR_INSIDE_MEDIA = True
# FACE_MARGIN_RATIO = 0.14
# SHOW_PREVIEW_RECTANGLE = True
# VERBOSE_LOG = True
# MAX_GPU_MEMORY_PERCENT = 95  # Skip det_set if usage would exceed this %
#
# CUSTOM_DET_SETS = [
#     # (640, 640), (704, 704), (768, 768), (800, 800), (896, 896),
#     # (1024, 1024), (1280, 1280), (1600, 1600), (2048, 2048)
#     (800, 800), (1024, 1024), (2048, 2048),
# ]
#
# face_apps = {}
# OUTPUT_DIRS = {}
# summary_counts = {}
#
# # ======================= MEMORY UTILITY ===========================
# def can_allocate_det_set(size):
#     try:
#         import pynvml
#         pynvml.nvmlInit()
#         handle = pynvml.nvmlDeviceGetHandleByIndex(0)
#         mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         used = mem_info.used / 1024**2
#         total = mem_info.total / 1024**2
#         percent_used = used / total * 100
#         pynvml.nvmlShutdown()
#         if percent_used > MAX_GPU_MEMORY_PERCENT:
#             if VERBOSE_LOG:
#                 print(f"[⚠️] Skipping {size} due to memory usage: {percent_used:.2f}%")
#             return False
#     except Exception as e:
#         print(f"[⚠️] pynvml error: {e}")
#     return True
#
# # ======================= SETUP ===========================
# def create_dirs(base_folder):
#     os.makedirs(base_folder, exist_ok=True)
#     for size in CUSTOM_DET_SETS:
#         name = f"{size[0]}x{size[1]}"
#         OUTPUT_DIRS[name] = os.path.join(base_folder, name)
#         os.makedirs(OUTPUT_DIRS[name], exist_ok=True)
#     OUTPUT_DIRS["bad"] = os.path.join(base_folder, "bad")
#     OUTPUT_DIRS["previews"] = os.path.join(base_folder, "previews")
#     os.makedirs(OUTPUT_DIRS["bad"], exist_ok=True)
#     os.makedirs(OUTPUT_DIRS["previews"], exist_ok=True)
#     return os.path.join(base_folder, "sorted_face_log.csv")
#
# def initialize_detectors():
#     for det_size in CUSTOM_DET_SETS:
#         if can_allocate_det_set(det_size):
#             app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
#             app.prepare(ctx_id=0, det_size=det_size)
#             face_apps[det_size] = app
#
# def get_sharpness_score(gray_img):
#     return cv2.Laplacian(gray_img, cv2.CV_64F).var()
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < 160 or h < 160:
#         hints.append("face too small")
#     if score < 0.6:
#         hints.append("very low score")
#     elif score < 0.75:
#         hints.append("low score")
#     elif score < 0.85:
#         hints.append("ok")
#     else:
#         hints.append("good")
#     if sharpness < 50:
#         hints.append("blurry")
#     elif sharpness < 100:
#         hints.append("soft")
#     else:
#         hints.append("sharp")
#     return "; ".join(hints)
#
# def round_det_set(face_w, face_h):
#     longest = max(face_w, face_h)
#     for size in CUSTOM_DET_SETS:
#         if longest <= size[0]:
#             return size
#     return CUSTOM_DET_SETS[-1]
#
# def process_image(image_path, writer_lock, writer):
#     img = cv2.imread(image_path)
#     if img is None:
#         return
#     fname = os.path.basename(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sharpness = round(get_sharpness_score(gray), 2)
#
#     face = None
#     chosen_size = None
#     if AUTO_ROUND_DETSET:
#         small_app = face_apps.get((640, 640))
#         if not small_app:
#             return
#         faces = small_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         if faces:
#             face = faces[0]
#             chosen_size = round_det_set(int(face.bbox[2] - face.bbox[0]), int(face.bbox[3] - face.bbox[1]))
#             app = face_apps.get(chosen_size)
#             if app:
#                 faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                 face = faces[0] if faces else None
#     else:
#         max_face_size = 0
#         for size in CUSTOM_DET_SETS:
#             app = face_apps.get(size)
#             if not app:
#                 continue
#             faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             if not faces:
#                 continue
#             f = faces[0]
#             w = int(f.bbox[2] - f.bbox[0])
#             h = int(f.bbox[3] - f.bbox[1])
#             if max(w, h) > max_face_size:
#                 max_face_size = max(w, h)
#                 face = f
#                 chosen_size = size
#
#     if face is None:
#         shutil.copy(image_path, os.path.join(OUTPUT_DIRS["bad"], fname))
#         with writer_lock:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", "no face detected"])
#         return
#
#     x1, y1, x2, y2 = map(int, face.bbox)
#     face_crop = img[y1:y2, x1:x2]
#     w, h = x2 - x1, y2 - y1
#     score = round(face.det_score, 4)
#     hint = generate_hint(w, h, score, sharpness)
#     det_name = f"{chosen_size[0]}x{chosen_size[1]}"
#
#     cv2.imwrite(os.path.join(OUTPUT_DIRS[det_name], fname), face_crop)
#
#     preview_img = img.copy()
#     if SHOW_PREVIEW_RECTANGLE:
#         cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     lines = [f"Score: {score}", f"Set: {det_name}", f"Size: {w}x{h}", f"Sharp: {sharpness}"]
#     for i, line in enumerate(lines):
#         cv2.putText(preview_img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     cv2.imwrite(os.path.join(OUTPUT_DIRS["previews"], fname), preview_img)
#
#     with writer_lock:
#         writer.writerow([fname, w, h, score, sharpness, det_name, hint])
#         summary_counts[det_name] = summary_counts.get(det_name, 0) + 1
#
# def main():
#     parser = argparse.ArgumentParser(description="Sort faces using InsightFace with dynamic GPU safety")
#     parser.add_argument("--input", required=True, help="Folder containing face images")
#     args = parser.parse_args()
#     input_folder = os.path.abspath(args.input)
#
#     timestamp = datetime.now().strftime("Sorted faces (%Y-%m-%d %H-%M-%S)")
#     base = os.path.join(settings.MEDIA_ROOT, "student_faces", timestamp) if SAVE_DIR_INSIDE_MEDIA else os.path.dirname(input_folder)
#     csv_path = create_dirs(base)
#     initialize_detectors()
#
#     images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#
#     lock = Lock()
#     with open(csv_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#         with ThreadPoolExecutor(max_workers=MAX_PARALLEL_IMAGES) as executor:
#             list(tqdm(executor.map(lambda p: process_image(p, lock, writer), images), total=len(images)))
#         writer.writerow([])
#         summary = ", ".join([f"{v} {k}" for k, v in summary_counts.items()])
#         writer.writerow(["SUMMARY", summary])
#         print(f"📊 SUMMARY: {summary}")
#         print(f"✅ Output saved to: {base}")
#
# if __name__ == "__main__":
#     main()
#











#
# #!/usr/bin/env python3
# # (** Mehdi _ with True not goog, with False no crash but gives all the low det_set results)Enhanced version with GPU memory error handling (safe fallback)
#
# import os
# import sys
# import cv2
# import csv
# import shutil
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# import argparse
# from concurrent.futures import ThreadPoolExecutor
# from threading import Lock
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# AUTO_ROUND_DETSET = False
# MAX_PARALLEL_IMAGES = 1
# SAVE_DIR_INSIDE_MEDIA = True
# FACE_MARGIN_RATIO = 0.14
# SHOW_PREVIEW_RECTANGLE = True
# VERBOSE_LOG = True
# CUSTOM_DET_SETS = [
#     (640, 640), (704, 704), (768, 768), (800, 800), (896, 896),
#     (1024, 1024), (1280, 1280), (1600, 1600), (2048, 2048)
# ]
#
# face_apps = {}
# OUTPUT_DIRS = {}
# summary_counts = {}
#
# def create_dirs(base_folder):
#     os.makedirs(base_folder, exist_ok=True)
#     for size in CUSTOM_DET_SETS:
#         name = f"{size[0]}x{size[1]}"
#         OUTPUT_DIRS[name] = os.path.join(base_folder, name)
#         os.makedirs(OUTPUT_DIRS[name], exist_ok=True)
#     OUTPUT_DIRS["bad"] = os.path.join(base_folder, "bad")
#     OUTPUT_DIRS["previews"] = os.path.join(base_folder, "previews")
#     os.makedirs(OUTPUT_DIRS["bad"], exist_ok=True)
#     os.makedirs(OUTPUT_DIRS["previews"], exist_ok=True)
#     return os.path.join(base_folder, "sorted_face_log.csv")
#
# def initialize_detectors():
#     for det_size in CUSTOM_DET_SETS:
#         try:
#             app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
#             app.prepare(ctx_id=0, det_size=det_size)
#             face_apps[det_size] = app
#         except Exception as e:
#             print(f"❌ Failed to load detector {det_size}: {e}")
#
# def get_sharpness_score(gray_img):
#     return cv2.Laplacian(gray_img, cv2.CV_64F).var()
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < 160 or h < 160:
#         hints.append("face too small")
#     if score < 0.6:
#         hints.append("very low score")
#     elif score < 0.75:
#         hints.append("low score")
#     elif score < 0.85:
#         hints.append("ok")
#     else:
#         hints.append("good")
#     if sharpness < 50:
#         hints.append("blurry")
#     elif sharpness < 100:
#         hints.append("soft")
#     else:
#         hints.append("sharp")
#     return "; ".join(hints)
#
# def round_det_set(face_w, face_h):
#     longest = max(face_w, face_h)
#     for size in CUSTOM_DET_SETS:
#         if longest <= size[0]:
#             return size
#     return CUSTOM_DET_SETS[-1]
#
# def process_image(image_path, writer_lock, writer):
#     img = cv2.imread(image_path)
#     if img is None:
#         return
#     fname = os.path.basename(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sharpness = round(get_sharpness_score(gray), 2)
#
#     face = None
#     chosen_size = None
#
#     if AUTO_ROUND_DETSET:
#         small_app = face_apps.get((640, 640))
#         if not small_app:
#             return
#         try:
#             faces = small_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         except:
#             faces = []
#         if faces:
#             face = faces[0]
#             chosen_size = round_det_set(int(face.bbox[2] - face.bbox[0]), int(face.bbox[3] - face.bbox[1]))
#             app = face_apps.get(chosen_size)
#             if app:
#                 try:
#                     faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                     face = faces[0] if faces else None
#                 except:
#                     face = None
#             else:
#                 face = None
#     else:
#         max_face_size = 0
#         for size in CUSTOM_DET_SETS:
#             app = face_apps.get(size)
#             if not app:
#                 continue
#             try:
#                 faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             except:
#                 continue
#             if not faces:
#                 continue
#             f = faces[0]
#             w = int(f.bbox[2] - f.bbox[0])
#             h = int(f.bbox[3] - f.bbox[1])
#             if max(w, h) > max_face_size:
#                 max_face_size = max(w, h)
#                 face = f
#                 chosen_size = size
#
#     if face is None:
#         shutil.copy(image_path, os.path.join(OUTPUT_DIRS["bad"], fname))
#         with writer_lock:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", "no face detected"])
#         return
#
#     x1, y1, x2, y2 = map(int, face.bbox)
#     face_crop = img[y1:y2, x1:x2]
#     w, h = x2 - x1, y2 - y1
#     score = round(face.det_score, 4)
#     hint = generate_hint(w, h, score, sharpness)
#     det_name = f"{chosen_size[0]}x{chosen_size[1]}"
#     cv2.imwrite(os.path.join(OUTPUT_DIRS[det_name], fname), face_crop)
#
#     preview_img = img.copy()
#     if SHOW_PREVIEW_RECTANGLE:
#         cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     lines = [f"Score: {score}", f"Set: {det_name}", f"Size: {w}x{h}", f"Sharp: {sharpness}"]
#     for i, line in enumerate(lines):
#         cv2.putText(preview_img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     cv2.imwrite(os.path.join(OUTPUT_DIRS["previews"], fname), preview_img)
#
#     with writer_lock:
#         writer.writerow([fname, w, h, score, sharpness, det_name, hint])
#         summary_counts[det_name] = summary_counts.get(det_name, 0) + 1
#
# def main():
#     parser = argparse.ArgumentParser(description="Sort faces using InsightFace")
#     parser.add_argument("--input", required=True, help="Folder with images")
#     args = parser.parse_args()
#     input_folder = os.path.abspath(args.input)
#
#     timestamp = datetime.now().strftime("Sorted faces (%Y-%m-%d %H-%M-%S)")
#     base = os.path.join(settings.MEDIA_ROOT, "student_faces", timestamp) if SAVE_DIR_INSIDE_MEDIA else os.path.dirname(input_folder)
#     csv_path = create_dirs(base)
#     initialize_detectors()
#
#     images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#
#     lock = Lock()
#     with open(csv_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#         with ThreadPoolExecutor(max_workers=MAX_PARALLEL_IMAGES) as executor:
#             list(tqdm(executor.map(lambda p: process_image(p, lock, writer), images), total=len(images)))
#         writer.writerow([])
#         summary = ", ".join([f"{v} {k}" for k, v in summary_counts.items()])
#         writer.writerow(["SUMMARY", summary])
#         print(f"📊 SUMMARY: {summary}")
#         print(f"✅ Output saved to: {base}")
#
# if __name__ == "__main__":
#     main()















# #!/usr/bin/env python3 (** Mehdi _ Seems goog tested on 14 images with AUTO_ROUND_DETSET = False and using a smaller set of det_set)
# import os
# import sys
# import cv2
# import csv
# import shutil
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# import argparse
# from concurrent.futures import ThreadPoolExecutor
# from threading import Lock
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # ======================= CONFIGURATION ===========================
# AUTO_ROUND_DETSET = False
# MAX_PARALLEL_IMAGES = 1
# SAVE_DIR_INSIDE_MEDIA = True
# FACE_MARGIN_RATIO = 0.14
# SHOW_PREVIEW_RECTANGLE = True
# VERBOSE_LOG = True
# MIN_CONFIDENCE_THRESHOLD = 0.70
# MIN_FACE_SIZE = 160  # pixels
#
# CUSTOM_DET_SETS = [
#     # (640, 640), (704, 704), (768, 768), (800, 800), (896, 896), (1024, 1024),
#     # (1280, 1280), (1600, 1600), (2048, 2048),
#     # (640, 640), (704, 704), (768, 768), (800, 800), (896, 896),
#     (800, 800), (1024, 1024), (1280, 1280), (1600, 1600),
# ]
#
# face_apps = {}
# OUTPUT_DIRS = {}
# summary_counts = {}
#
# def create_dirs(base_folder):
#     os.makedirs(base_folder, exist_ok=True)
#     for size in CUSTOM_DET_SETS:
#         name = f"{size[0]}x{size[1]}"
#         OUTPUT_DIRS[name] = os.path.join(base_folder, name)
#         os.makedirs(OUTPUT_DIRS[name], exist_ok=True)
#     OUTPUT_DIRS["bad"] = os.path.join(base_folder, "bad")
#     OUTPUT_DIRS["previews"] = os.path.join(base_folder, "previews")
#     os.makedirs(OUTPUT_DIRS["bad"], exist_ok=True)
#     os.makedirs(OUTPUT_DIRS["previews"], exist_ok=True)
#     return os.path.join(base_folder, "sorted_face_log.csv")
#
# def initialize_detectors():
#     for det_size in CUSTOM_DET_SETS:
#         app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
#         app.prepare(ctx_id=0, det_size=det_size)
#         face_apps[det_size] = app
#
# def get_sharpness_score(gray_img):
#     return cv2.Laplacian(gray_img, cv2.CV_64F).var()
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
#         hints.append("face too small")
#     if score < 0.6:
#         hints.append("very low score")
#     elif score < 0.75:
#         hints.append("low score")
#     elif score < 0.85:
#         hints.append("ok")
#     else:
#         hints.append("good")
#     if sharpness < 50:
#         hints.append("blurry")
#     elif sharpness < 100:
#         hints.append("soft")
#     else:
#         hints.append("sharp")
#     return "; ".join(hints)
#
# def round_det_set(face_w, face_h):
#     longest = max(face_w, face_h)
#     for size in CUSTOM_DET_SETS:
#         if longest <= size[0]:
#             return size
#     return CUSTOM_DET_SETS[-1]
#
# def process_image(image_path, writer_lock, writer):
#     img = cv2.imread(image_path)
#     if img is None:
#         return
#     fname = os.path.basename(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sharpness = round(get_sharpness_score(gray), 2)
#
#     face = None
#     chosen_size = None
#     if AUTO_ROUND_DETSET:
#         small_app = face_apps[(640, 640)]
#         faces = small_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         if faces:
#             face = faces[0]
#             chosen_size = round_det_set(int(face.bbox[2] - face.bbox[0]), int(face.bbox[3] - face.bbox[1]))
#             app = face_apps[chosen_size]
#             faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             face = faces[0] if faces else None
#         else:
#             face = None
#     else:
#         best_score = -1
#         best_metric = -1
#         for size in CUSTOM_DET_SETS:
#             app = face_apps[size]
#             try:
#                 faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             except:
#                 continue
#             if not faces:
#                 continue
#             f = faces[0]
#             w = int(f.bbox[2] - f.bbox[0])
#             h = int(f.bbox[3] - f.bbox[1])
#             if f.det_score < MIN_CONFIDENCE_THRESHOLD:
#                 continue
#             if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
#                 continue
#             metric = f.det_score * max(w, h)
#             if metric > best_metric:
#                 best_metric = metric
#                 best_score = f.det_score
#                 face = f
#                 chosen_size = size
#
#     if face is None:
#         shutil.copy(image_path, os.path.join(OUTPUT_DIRS["bad"], fname))
#         with writer_lock:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", "no face detected"])
#         return
#
#     x1, y1, x2, y2 = map(int, face.bbox)
#     face_crop = img[y1:y2, x1:x2]
#     w, h = x2 - x1, y2 - y1
#     score = round(face.det_score, 4)
#     hint = generate_hint(w, h, score, sharpness)
#     det_name = f"{chosen_size[0]}x{chosen_size[1]}"
#
#     # Save cropped
#     cv2.imwrite(os.path.join(OUTPUT_DIRS[det_name], fname), face_crop)
#
#     # Save preview
#     preview_img = img.copy()
#     if SHOW_PREVIEW_RECTANGLE:
#         cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     lines = [f"Score: {score}", f"Set: {det_name}", f"Size: {w}x{h}", f"Sharp: {sharpness}"]
#     for i, line in enumerate(lines):
#         cv2.putText(preview_img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     cv2.imwrite(os.path.join(OUTPUT_DIRS["previews"], fname), preview_img)
#
#     with writer_lock:
#         writer.writerow([fname, w, h, score, sharpness, det_name, hint])
#         summary_counts[det_name] = summary_counts.get(det_name, 0) + 1
#
# def main():
#     parser = argparse.ArgumentParser(description="Sort faces using InsightFace (dual-mode: full scan or auto-det_set)")
#     parser.add_argument("--input", required=True, help="Folder containing face images")
#     args = parser.parse_args()
#     input_folder = os.path.abspath(args.input)
#
#     timestamp = datetime.now().strftime("Sorted faces (%Y-%m-%d %H-%M-%S)")
#     base = os.path.join(settings.MEDIA_ROOT, "student_faces", timestamp) if SAVE_DIR_INSIDE_MEDIA else os.path.dirname(input_folder)
#     csv_path = create_dirs(base)
#     initialize_detectors()
#
#     images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#
#     lock = Lock()
#     with open(csv_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#         with ThreadPoolExecutor(max_workers=MAX_PARALLEL_IMAGES) as executor:
#             list(tqdm(executor.map(lambda p: process_image(p, lock, writer), images), total=len(images)))
#         writer.writerow([])
#         summary = ", ".join([f"{v} {k}" for k, v in summary_counts.items()])
#         writer.writerow(["SUMMARY", summary])
#         print(f"📊 SUMMARY: {summary}")
#         print(f"✅ Output saved to: {base}")
#
# if __name__ == "__main__":
#     main()

















# #!/usr/bin/env python3
# # Final script with memory control for face sorting using InsightFace
#
# import os
# import sys
# import cv2
# import csv
# import shutil
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# import argparse
# from concurrent.futures import ThreadPoolExecutor
# from threading import Lock
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # ======================= CONFIGURATION ===========================
# AUTO_ROUND_DETSET = False
# MAX_PARALLEL_IMAGES = 1
# SAVE_DIR_INSIDE_MEDIA = True
# FACE_MARGIN_RATIO = 0.14
# SHOW_PREVIEW_RECTANGLE = True
# VERBOSE_LOG = True
#
# CUSTOM_DET_SETS = [
#     (640, 640), (704, 704), (768, 768), (800, 800), (896, 896), (1024, 1024),
#     # Add higher resolutions cautiously based on GPU memory
#     (1280, 1280), (1600, 1600), (2048, 2048)
# ]
#
# face_apps = {}
# OUTPUT_DIRS = {}
# summary_counts = {}
#
# def create_dirs(base_folder):
#     os.makedirs(base_folder, exist_ok=True)
#     for size in CUSTOM_DET_SETS:
#         name = f"{size[0]}x{size[1]}"
#         OUTPUT_DIRS[name] = os.path.join(base_folder, name)
#         os.makedirs(OUTPUT_DIRS[name], exist_ok=True)
#     OUTPUT_DIRS["bad"] = os.path.join(base_folder, "bad")
#     OUTPUT_DIRS["previews"] = os.path.join(base_folder, "previews")
#     os.makedirs(OUTPUT_DIRS["bad"], exist_ok=True)
#     os.makedirs(OUTPUT_DIRS["previews"], exist_ok=True)
#     return os.path.join(base_folder, "sorted_face_log.csv")
#
# def initialize_detectors():
#     for det_size in CUSTOM_DET_SETS:
#         try:
#             app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
#             app.prepare(ctx_id=0, det_size=det_size)
#             face_apps[det_size] = app
#         except Exception as e:
#             print(f"⚠️ Skipping det_set {det_size}: {e}")
#
# def get_sharpness_score(gray_img):
#     return cv2.Laplacian(gray_img, cv2.CV_64F).var()
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < 160 or h < 160:
#         hints.append("face too small")
#     if score < 0.6:
#         hints.append("very low score")
#     elif score < 0.75:
#         hints.append("low score")
#     elif score < 0.85:
#         hints.append("ok")
#     else:
#         hints.append("good")
#     if sharpness < 50:
#         hints.append("blurry")
#     elif sharpness < 100:
#         hints.append("soft")
#     else:
#         hints.append("sharp")
#     return "; ".join(hints)
#
# def round_det_set(face_w, face_h):
#     longest = max(face_w, face_h)
#     for size in CUSTOM_DET_SETS:
#         if longest <= size[0]:
#             return size
#     return CUSTOM_DET_SETS[-1]
#
# def process_image(image_path, writer_lock, writer):
#     img = cv2.imread(image_path)
#     if img is None:
#         return
#     fname = os.path.basename(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sharpness = round(get_sharpness_score(gray), 2)
#
#     face = None
#     chosen_size = None
#     if AUTO_ROUND_DETSET:
#         small_app = face_apps.get((640, 640))
#         if small_app:
#             faces = small_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             if faces:
#                 face = faces[0]
#                 chosen_size = round_det_set(int(face.bbox[2] - face.bbox[0]), int(face.bbox[3] - face.bbox[1]))
#                 app = face_apps.get(chosen_size)
#                 if app:
#                     faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                     face = faces[0] if faces else None
#             else:
#                 face = None
#     else:
#         max_face_size = 0
#         for size, app in face_apps.items():
#             try:
#                 faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#                 if not faces:
#                     continue
#                 f = faces[0]
#                 w = int(f.bbox[2] - f.bbox[0])
#                 h = int(f.bbox[3] - f.bbox[1])
#                 if max(w, h) > max_face_size:
#                     max_face_size = max(w, h)
#                     face = f
#                     chosen_size = size
#             except Exception as e:
#                 continue
#
#     if face is None:
#         shutil.copy(image_path, os.path.join(OUTPUT_DIRS["bad"], fname))
#         with writer_lock:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", "no face detected"])
#         return
#
#     x1, y1, x2, y2 = map(int, face.bbox)
#     face_crop = img[y1:y2, x1:x2]
#     w, h = x2 - x1, y2 - y1
#     score = round(face.det_score, 4)
#     hint = generate_hint(w, h, score, sharpness)
#     det_name = f"{chosen_size[0]}x{chosen_size[1]}"
#
#     cv2.imwrite(os.path.join(OUTPUT_DIRS[det_name], fname), face_crop)
#
#     preview_img = img.copy()
#     if SHOW_PREVIEW_RECTANGLE:
#         cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     lines = [f"Score: {score}", f"Set: {det_name}", f"Size: {w}x{h}", f"Sharp: {sharpness}"]
#     for i, line in enumerate(lines):
#         cv2.putText(preview_img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     cv2.imwrite(os.path.join(OUTPUT_DIRS["previews"], fname), preview_img)
#
#     with writer_lock:
#         writer.writerow([fname, w, h, score, sharpness, det_name, hint])
#         summary_counts[det_name] = summary_counts.get(det_name, 0) + 1
#
# def main():
#     parser = argparse.ArgumentParser(description="Sort faces using InsightFace (dual-mode: full scan or auto-det_set)")
#     parser.add_argument("--input", required=True, help="Folder containing face images")
#     args = parser.parse_args()
#     input_folder = os.path.abspath(args.input)
#
#     timestamp = datetime.now().strftime("Sorted faces (%Y-%m-%d %H-%M-%S)")
#     base = os.path.join(settings.MEDIA_ROOT, "student_faces", timestamp) if SAVE_DIR_INSIDE_MEDIA else os.path.dirname(input_folder)
#     csv_path = create_dirs(base)
#     initialize_detectors()
#
#     images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#
#     lock = Lock()
#     with open(csv_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#         with ThreadPoolExecutor(max_workers=MAX_PARALLEL_IMAGES) as executor:
#             list(tqdm(executor.map(lambda p: process_image(p, lock, writer), images), total=len(images)))
#         writer.writerow([])
#         summary = ", ".join([f"{v} {k}" for k, v in summary_counts.items()])
#         writer.writerow(["SUMMARY", summary])
#         print(f"📊 SUMMARY: {summary}")
#         print(f"✅ Output saved to: {base}")
#
# if __name__ == "__main__":
#     main()
#
















# import argparse
# import os
# import sys
# import cv2
# import csv
# import shutil
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # ============================ CONFIGURATION =============================
# GPU_ONLY = True
# MAX_PARALLEL_IMAGES = 1
# AUTO_ROUND_DETSET = False
# CUSTOM_DET_SETS = [
#     (640, 640), (704, 704), (768, 768), (800, 800), (896, 896),
#     # (1024, 1024), (1280, 1280), (1600, 1600), (2048, 2048)
# ]
# SAVE_DIR_INSIDE_MEDIA = True
# FACE_MARGIN_RATIO = 0.14
# FACE_MARGIN_RATIO_PREVIEW = 0.06
# PREVIEW_BOX_CROP_RATIO = 0.85
# SHOW_PREVIEW_RECTANGLE = True
# VERBOSE_LOG = True
#
# # ============================ DET SET HELPERS =============================
# def round_to_nearest_detset(w, h, detsets):
#     size = max(w, h)
#     sizes = [max(s) for s in detsets]
#     closest = min(sizes, key=lambda x: abs(x - size))
#     return f"{closest}x{closest}"
#
# # ============================ INIT OUTPUT =============================
# now = datetime.now().strftime("Sorted faces (%Y-%m-%d %H-%M-%S)")
# BASE = os.path.join(settings.MEDIA_ROOT, "student_faces", now) if SAVE_DIR_INSIDE_MEDIA else None
# OUTPUT_DIRS = {}
#
# def create_dirs(base_folder):
#     os.makedirs(base_folder, exist_ok=True)
#     for size in CUSTOM_DET_SETS:
#         name = f"{size[0]}x{size[1]}"
#         OUTPUT_DIRS[name] = os.path.join(base_folder, name)
#         os.makedirs(OUTPUT_DIRS[name], exist_ok=True)
#     OUTPUT_DIRS["bad"] = os.path.join(base_folder, "bad")
#     OUTPUT_DIRS["previews"] = os.path.join(base_folder, "previews")
#     os.makedirs(OUTPUT_DIRS["bad"], exist_ok=True)
#     os.makedirs(OUTPUT_DIRS["previews"], exist_ok=True)
#     return os.path.join(base_folder, "sorted_face_log.csv")
#
# # ============================ METRICS =============================
# def get_sharpness_score(gray_img):
#     return cv2.Laplacian(gray_img, cv2.CV_64F).var()
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < 160 or h < 160:
#         hints.append("face too small")
#     if score < 0.6:
#         hints.append("very low score")
#     elif score < 0.75:
#         hints.append("low score")
#     elif score < 0.85:
#         hints.append("ok")
#     else:
#         hints.append("good")
#
#     if sharpness < 50:
#         hints.append("blurry")
#     elif sharpness < 100:
#         hints.append("soft")
#     else:
#         hints.append("sharp")
#     return "; ".join(hints)
#
# # ============================ FACE ANALYSIS =============================
# face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# face_app.prepare(ctx_id=0, det_size=(2048, 2048))  # max size for safety
#
# summary_counts = {}
#
# # ============================ IMAGE PROCESS =============================
# def process_image(image_path, writer_lock, writer):
#     img = cv2.imread(image_path)
#     if img is None:
#         return
#     fname = os.path.basename(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sharpness = round(get_sharpness_score(gray), 2)
#
#     faces = face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
#     if not faces:
#         out_path = os.path.join(OUTPUT_DIRS["bad"], fname)
#         shutil.copy(image_path, out_path)
#         hint = "no face detected"
#         with writer_lock:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", hint])
#         return
#
#     face = faces[0]
#     x1, y1, x2, y2 = map(int, face.bbox)
#     face_crop = img[y1:y2, x1:x2]
#     w, h = x2 - x1, y2 - y1
#     score = round(face.det_score, 4)
#
#     if AUTO_ROUND_DETSET:
#         det_set = round_to_nearest_detset(w, h, CUSTOM_DET_SETS)
#     else:
#         det_set = f"{CUSTOM_DET_SETS[-1][0]}x{CUSTOM_DET_SETS[-1][1]}"
#         for s in CUSTOM_DET_SETS:
#             if max(w, h) <= max(s):
#                 det_set = f"{s[0]}x{s[1]}"
#                 break
#
#     hint = generate_hint(w, h, score, sharpness)
#     save_path = os.path.join(OUTPUT_DIRS[det_set], fname)
#     cv2.imwrite(save_path, face_crop)
#
#     # Save preview
#     preview_img = img.copy()
#     if SHOW_PREVIEW_RECTANGLE:
#         cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     overlay_texts = [
#         f"Score: {score}",
#         f"Set: {det_set}",
#         f"Size: {w}x{h}",
#         f"Sharp: {sharpness}"
#     ]
#     for i, line in enumerate(overlay_texts):
#         cv2.putText(preview_img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     preview_path = os.path.join(OUTPUT_DIRS["previews"], fname)
#     cv2.imwrite(preview_path, preview_img)
#
#     # Write CSV
#     with writer_lock:
#         writer.writerow([fname, w, h, score, sharpness, det_set, hint])
#         summary_counts[det_set] = summary_counts.get(det_set, 0) + 1
#
# # ============================ MAIN =============================
# def main():
#     from threading import Lock
#     parser = argparse.ArgumentParser(description="Sort faces dynamically by detection set")
#     parser.add_argument("--input", required=True, help="Path to folder containing input images")
#     args = parser.parse_args()
#     input_folder = os.path.abspath(args.input)
#     images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#     csv_path = create_dirs(BASE)
#
#     lock = Lock()
#     with open(csv_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#
#         with ThreadPoolExecutor(max_workers=MAX_PARALLEL_IMAGES) as executor:
#             list(tqdm(executor.map(lambda p: process_image(p, lock, writer), images), total=len(images)))
#
#         writer.writerow([])
#         summary_text = ", ".join([f"{v} {k}" for k, v in summary_counts.items()])
#         writer.writerow(["SUMMARY", summary_text])
#         print(f"📊 SUMMARY: {summary_text}")
#         print(f"✅ Output saved to: {BASE}")
#
# if __name__ == "__main__":
#     main()








# #!/usr/bin/env python3 (** Mehdi _ Good if I turn off the AUTO_ROUND_DETESET)
# import os
# import sys
# import cv2
# import csv
# import shutil
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# import argparse
# from concurrent.futures import ThreadPoolExecutor
# from threading import Lock
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # ======================= CONFIGURATION ===========================
# AUTO_ROUND_DETSET = False
# MAX_PARALLEL_IMAGES = 1
# SAVE_DIR_INSIDE_MEDIA = True
# FACE_MARGIN_RATIO = 0.14
# SHOW_PREVIEW_RECTANGLE = True
# VERBOSE_LOG = True
#
# CUSTOM_DET_SETS = [
#     (640, 640), (704, 704), (768, 768), (800, 800), (896, 896), (1024, 1024), (1280, 1280)
#     # (1024, 1024), (1280, 1280), (1600, 1600), (2048, 2048)
# ]
#
# face_apps = {}
# OUTPUT_DIRS = {}
# summary_counts = {}
#
# def create_dirs(base_folder):
#     os.makedirs(base_folder, exist_ok=True)
#     for size in CUSTOM_DET_SETS:
#         name = f"{size[0]}x{size[1]}"
#         OUTPUT_DIRS[name] = os.path.join(base_folder, name)
#         os.makedirs(OUTPUT_DIRS[name], exist_ok=True)
#     OUTPUT_DIRS["bad"] = os.path.join(base_folder, "bad")
#     OUTPUT_DIRS["previews"] = os.path.join(base_folder, "previews")
#     os.makedirs(OUTPUT_DIRS["bad"], exist_ok=True)
#     os.makedirs(OUTPUT_DIRS["previews"], exist_ok=True)
#     return os.path.join(base_folder, "sorted_face_log.csv")
#
# def initialize_detectors():
#     for det_size in CUSTOM_DET_SETS:
#         app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
#         app.prepare(ctx_id=0, det_size=det_size)
#         face_apps[det_size] = app
#
# def get_sharpness_score(gray_img):
#     return cv2.Laplacian(gray_img, cv2.CV_64F).var()
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < 160 or h < 160:
#         hints.append("face too small")
#     if score < 0.6:
#         hints.append("very low score")
#     elif score < 0.75:
#         hints.append("low score")
#     elif score < 0.85:
#         hints.append("ok")
#     else:
#         hints.append("good")
#     if sharpness < 50:
#         hints.append("blurry")
#     elif sharpness < 100:
#         hints.append("soft")
#     else:
#         hints.append("sharp")
#     return "; ".join(hints)
#
# def round_det_set(face_w, face_h):
#     longest = max(face_w, face_h)
#     for size in CUSTOM_DET_SETS:
#         if longest <= size[0]:
#             return size
#     return CUSTOM_DET_SETS[-1]
#
# def process_image(image_path, writer_lock, writer):
#     img = cv2.imread(image_path)
#     if img is None:
#         return
#     fname = os.path.basename(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sharpness = round(get_sharpness_score(gray), 2)
#
#     face = None
#     chosen_size = None
#     if AUTO_ROUND_DETSET:
#         small_app = face_apps[(640, 640)]
#         faces = small_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#         if faces:
#             face = faces[0]
#             chosen_size = round_det_set(int(face.bbox[2] - face.bbox[0]), int(face.bbox[3] - face.bbox[1]))
#             app = face_apps[chosen_size]
#             faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             face = faces[0] if faces else None
#         else:
#             face = None
#     else:
#         max_face_size = 0
#         for size in CUSTOM_DET_SETS:
#             app = face_apps[size]
#             faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             if not faces:
#                 continue
#             f = faces[0]
#             w = int(f.bbox[2] - f.bbox[0])
#             h = int(f.bbox[3] - f.bbox[1])
#             if max(w, h) > max_face_size:
#                 max_face_size = max(w, h)
#                 face = f
#                 chosen_size = size
#
#     if face is None:
#         shutil.copy(image_path, os.path.join(OUTPUT_DIRS["bad"], fname))
#         with writer_lock:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", "no face detected"])
#         return
#
#     x1, y1, x2, y2 = map(int, face.bbox)
#     face_crop = img[y1:y2, x1:x2]
#     w, h = x2 - x1, y2 - y1
#     score = round(face.det_score, 4)
#     hint = generate_hint(w, h, score, sharpness)
#     det_name = f"{chosen_size[0]}x{chosen_size[1]}"
#
#     # Save cropped
#     cv2.imwrite(os.path.join(OUTPUT_DIRS[det_name], fname), face_crop)
#
#     # Save preview
#     preview_img = img.copy()
#     if SHOW_PREVIEW_RECTANGLE:
#         cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     lines = [f"Score: {score}", f"Set: {det_name}", f"Size: {w}x{h}", f"Sharp: {sharpness}"]
#     for i, line in enumerate(lines):
#         cv2.putText(preview_img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     cv2.imwrite(os.path.join(OUTPUT_DIRS["previews"], fname), preview_img)
#
#     with writer_lock:
#         writer.writerow([fname, w, h, score, sharpness, det_name, hint])
#         summary_counts[det_name] = summary_counts.get(det_name, 0) + 1
#
# def main():
#     parser = argparse.ArgumentParser(description="Sort faces using InsightFace (dual-mode: full scan or auto-det_set)")
#     parser.add_argument("--input", required=True, help="Folder containing face images")
#     args = parser.parse_args()
#     input_folder = os.path.abspath(args.input)
#
#     timestamp = datetime.now().strftime("Sorted faces (%Y-%m-%d %H-%M-%S)")
#     base = os.path.join(settings.MEDIA_ROOT, "student_faces", timestamp) if SAVE_DIR_INSIDE_MEDIA else os.path.dirname(input_folder)
#     csv_path = create_dirs(base)
#     initialize_detectors()
#
#     images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#
#     lock = Lock()
#     with open(csv_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#         with ThreadPoolExecutor(max_workers=MAX_PARALLEL_IMAGES) as executor:
#             list(tqdm(executor.map(lambda p: process_image(p, lock, writer), images), total=len(images)))
#         writer.writerow([])
#         summary = ", ".join([f"{v} {k}" for k, v in summary_counts.items()])
#         writer.writerow(["SUMMARY", summary])
#         print(f"📊 SUMMARY: {summary}")
#         print(f"✅ Output saved to: {base}")
#
# if __name__ == "__main__":
#     main()












# #!/usr/bin/env python3   (Mehdi _ Good but a little slow)
# import os
# import sys
# import cv2
# import csv
# import shutil
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# import argparse
# from concurrent.futures import ThreadPoolExecutor
# from threading import Lock
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # ======================= CONFIGURATION ===========================
# GPU_ONLY = True
# MAX_PARALLEL_IMAGES = 1
# SAVE_DIR_INSIDE_MEDIA = True
# FACE_MARGIN_RATIO = 0.14
# FACE_MARGIN_RATIO_PREVIEW = 0.06
# PREVIEW_BOX_CROP_RATIO = 0.85
# SHOW_PREVIEW_RECTANGLE = True
# VERBOSE_LOG = True
#
# # Detection sets we want to support per image
# DETECTION_SETS = [
#     (640, 640),
#     (800, 800),
#     (1024, 1024),
#     (1280, 1280),
#     (1600, 1600),
#     (2048, 2048),
# ]
#
# # Prepare output folder
# now = datetime.now().strftime("Sorted faces (%Y-%m-%d %H-%M-%S)")
# BASE = None
# OUTPUT_DIRS = {}
# face_apps = {}
#
# def create_dirs(base_folder):
#     os.makedirs(base_folder, exist_ok=True)
#     for size in DETECTION_SETS:
#         name = f"{size[0]}x{size[1]}"
#         OUTPUT_DIRS[name] = os.path.join(base_folder, name)
#         os.makedirs(OUTPUT_DIRS[name], exist_ok=True)
#     OUTPUT_DIRS["bad"] = os.path.join(base_folder, "bad")
#     OUTPUT_DIRS["previews"] = os.path.join(base_folder, "previews")
#     os.makedirs(OUTPUT_DIRS["bad"], exist_ok=True)
#     os.makedirs(OUTPUT_DIRS["previews"], exist_ok=True)
#     return os.path.join(base_folder, "sorted_face_log.csv")
#
# def initialize_detectors():
#     global face_apps
#     for det_size in DETECTION_SETS:
#         app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider"])
#         app.prepare(ctx_id=0, det_size=det_size)
#         face_apps[det_size] = app
#
# # =============== UTILITIES ==================
# def get_sharpness_score(gray_img):
#     return cv2.Laplacian(gray_img, cv2.CV_64F).var()
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < 160 or h < 160:
#         hints.append("face too small")
#     if score < 0.6:
#         hints.append("very low score")
#     elif score < 0.75:
#         hints.append("low score")
#     elif score < 0.85:
#         hints.append("ok")
#     else:
#         hints.append("good")
#     if sharpness < 50:
#         hints.append("blurry")
#     elif sharpness < 100:
#         hints.append("soft")
#     else:
#         hints.append("sharp")
#     return "; ".join(hints)
#
# def choose_best_face_app(img):
#     max_face_size = 0
#     best_app = None
#     best_face = None
#     best_size = None
#
#     for size in DETECTION_SETS:
#         app = face_apps[size]
#         try:
#             faces = app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             if not faces:
#                 continue
#             face = faces[0]
#             x1, y1, x2, y2 = map(int, face.bbox)
#             w, h = x2 - x1, y2 - y1
#             if max(w, h) > max_face_size:
#                 max_face_size = max(w, h)
#                 best_app = app
#                 best_face = face
#                 best_size = size
#         except Exception:
#             continue
#
#     return best_app, best_face, best_size
#
# # =============== PROCESS IMAGE ===============
# summary_counts = {}
#
# def process_image(image_path, writer_lock, writer):
#     img = cv2.imread(image_path)
#     if img is None:
#         return
#     fname = os.path.basename(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sharpness = round(get_sharpness_score(gray), 2)
#
#     app, face, size = choose_best_face_app(img)
#     if face is None:
#         out_path = os.path.join(OUTPUT_DIRS["bad"], fname)
#         shutil.copy(image_path, out_path)
#         hint = "no face detected"
#         with writer_lock:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", hint])
#         return
#
#     x1, y1, x2, y2 = map(int, face.bbox)
#     face_crop = img[y1:y2, x1:x2]
#     w, h = x2 - x1, y2 - y1
#     score = round(face.det_score, 4)
#     hint = generate_hint(w, h, score, sharpness)
#     det_name = f"{size[0]}x{size[1]}"
#
#     # Save cropped
#     save_path = os.path.join(OUTPUT_DIRS[det_name], fname)
#     cv2.imwrite(save_path, face_crop)
#
#     # Save preview
#     preview_img = img.copy()
#     if SHOW_PREVIEW_RECTANGLE:
#         cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     overlay_texts = [
#         f"Score: {score}",
#         f"Set: {det_name}",
#         f"Size: {w}x{h}",
#         f"Sharp: {sharpness}"
#     ]
#     for i, line in enumerate(overlay_texts):
#         cv2.putText(preview_img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     preview_path = os.path.join(OUTPUT_DIRS["previews"], fname)
#     cv2.imwrite(preview_path, preview_img)
#
#     # Write CSV
#     with writer_lock:
#         writer.writerow([fname, w, h, score, sharpness, det_name, hint])
#         summary_counts[det_name] = summary_counts.get(det_name, 0) + 1
#
# # =============== MAIN =======================
# def main():
#     global BASE
#     parser = argparse.ArgumentParser(description="Sort faces dynamically per best det_set")
#     parser.add_argument("--input", required=True, help="Path to folder containing input images")
#     args = parser.parse_args()
#
#     input_folder = os.path.abspath(args.input)
#     BASE = os.path.join(settings.MEDIA_ROOT, "student_faces", now) if SAVE_DIR_INSIDE_MEDIA else os.path.dirname(input_folder)
#     csv_path = create_dirs(BASE)
#     initialize_detectors()
#
#     all_images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#
#     lock = Lock()
#     with open(csv_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#
#         with ThreadPoolExecutor(max_workers=MAX_PARALLEL_IMAGES) as executor:
#             list(tqdm(executor.map(lambda p: process_image(p, lock, writer), all_images), total=len(all_images)))
#
#         # Add summary
#         writer.writerow([])
#         summary_text = ", ".join([f"{v} {k}" for k, v in summary_counts.items()])
#         writer.writerow(["SUMMARY", summary_text])
#         print(f"📊 SUMMARY: {summary_text}")
#         print(f"✅ Sorting complete. Output saved to: {BASE}")
#
# if __name__ == "__main__":
#     main()
#









# import argparse
# import os
# import sys
# import cv2
# import csv
# import shutil
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # ============================ CONFIGURATION =============================
# GPU_ONLY = True
# MAX_PARALLEL_IMAGES = 1  # Scale this based on GPU strength
# DYNAMIC_DET_SET = True
# SAVE_DIR_INSIDE_MEDIA = True  # If False, save in selected input folder
# FACE_MARGIN_RATIO = 0.14
# FACE_MARGIN_RATIO_PREVIEW = 0.06
# PREVIEW_BOX_CROP_RATIO = 0.85
# SHOW_PREVIEW_RECTANGLE = True
# VERBOSE_LOG = True
#
# DETECTION_SETS = [
#     (640, 640),
#     (800, 800),
#     (1024, 1024),
#     (1280, 1280),
#     (1600, 1600),
#     (2048, 2048),
# ]
#
# # ============================ PREPARE OUTPUT =============================
# now = datetime.now().strftime("Sorted faces (%Y-%m-%d %H-%M-%S)")
# BASE = os.path.join(settings.MEDIA_ROOT, "student_faces", now) if SAVE_DIR_INSIDE_MEDIA else None
# OUTPUT_DIRS = {}
#
# def create_dirs(base_folder):
#     os.makedirs(base_folder, exist_ok=True)
#     for size in DETECTION_SETS:
#         name = f"{size[0]}x{size[1]}"
#         OUTPUT_DIRS[name] = os.path.join(base_folder, name)
#         os.makedirs(OUTPUT_DIRS[name], exist_ok=True)
#     OUTPUT_DIRS["bad"] = os.path.join(base_folder, "bad")
#     OUTPUT_DIRS["previews"] = os.path.join(base_folder, "previews")
#     os.makedirs(OUTPUT_DIRS["bad"], exist_ok=True)
#     os.makedirs(OUTPUT_DIRS["previews"], exist_ok=True)
#     return os.path.join(base_folder, "sorted_face_log.csv")
#
# # ============================ ANALYSIS UTILS =============================
# def get_sharpness_score(gray_img):
#     return cv2.Laplacian(gray_img, cv2.CV_64F).var()
#
# def recommended_det_set(face_w, face_h):
#     size = max(face_w, face_h)
#     for w, h in DETECTION_SETS:
#         if size <= w:
#             return f"{w}x{h}"
#     return f"{DETECTION_SETS[-1][0]}x{DETECTION_SETS[-1][1]}"
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < 160 or h < 160:
#         hints.append("face too small")
#     if score < 0.6:
#         hints.append("very low score")
#     elif score < 0.75:
#         hints.append("low score")
#     elif score < 0.85:
#         hints.append("ok")
#     else:
#         hints.append("good")
#
#     if sharpness < 50:
#         hints.append("blurry")
#     elif sharpness < 100:
#         hints.append("soft")
#     else:
#         hints.append("sharp")
#     return "; ".join(hints)
#
# # ============================ FACE DETECTOR =============================
# face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# face_app.prepare(ctx_id=0, det_size=(2048, 2048))  # default max size
#
# # ============================ PROCESS FUNCTION =============================
# summary_counts = {}
#
# def process_image(image_path, writer_lock, writer):
#     img = cv2.imread(image_path)
#     if img is None:
#         return
#     fname = os.path.basename(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sharpness = round(get_sharpness_score(gray), 2)
#
#     faces = face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
#     if not faces:
#         out_path = os.path.join(OUTPUT_DIRS["bad"], fname)
#         shutil.copy(image_path, out_path)
#         hint = "no face detected"
#         with writer_lock:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", hint])
#         return
#
#     face = faces[0]
#     x1, y1, x2, y2 = map(int, face.bbox)
#     face_crop = img[y1:y2, x1:x2]
#     w, h = x2 - x1, y2 - y1
#     score = round(face.det_score, 4)
#
#     det_set = recommended_det_set(w, h)
#     hint = generate_hint(w, h, score, sharpness)
#
#     # Save cropped
#     save_path = os.path.join(OUTPUT_DIRS[det_set], fname)
#     cv2.imwrite(save_path, face_crop)
#
#     # Save preview
#     preview_img = img.copy()
#     if SHOW_PREVIEW_RECTANGLE:
#         cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     overlay_texts = [
#         f"Score: {score}",
#         f"Set: {det_set}",
#         f"Size: {w}x{h}",
#         f"Sharp: {sharpness}"
#     ]
#     for i, line in enumerate(overlay_texts):
#         cv2.putText(preview_img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     preview_path = os.path.join(OUTPUT_DIRS["previews"], fname)
#     cv2.imwrite(preview_path, preview_img)
#
#     # Write CSV
#     with writer_lock:
#         writer.writerow([fname, w, h, score, sharpness, det_set, hint])
#         summary_counts[det_set] = summary_counts.get(det_set, 0) + 1
#
# # ============================ MAIN ========================================
# def main():
#     from threading import Lock
#     parser = argparse.ArgumentParser(description="Sort faces dynamically by detection set")
#     parser.add_argument("--input", required=True, help="Path to folder containing input images")
#     args = parser.parse_args()
#     input_folder = os.path.abspath(args.input)
#     all_images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#     csv_path = create_dirs(BASE)
#
#     lock = Lock()
#     with open(csv_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#
#         with ThreadPoolExecutor(max_workers=MAX_PARALLEL_IMAGES) as executor:
#             list(tqdm(executor.map(lambda p: process_image(p, lock, writer), all_images), total=len(all_images)))
#
#         # Add summary
#         writer.writerow([])
#         summary_text = ", ".join([f"{v} {k}" for k, v in summary_counts.items()])
#         writer.writerow(["SUMMARY", summary_text])
#         print(f"📊 SUMMARY: {summary_text}")
#         print(f"✅ Sorting complete. Output saved to: {BASE}")
#
# if __name__ == "__main__":
#     main()










# import os
# import sys
# import cv2
# import csv
# import shutil
# import numpy as np
# from tqdm import tqdm
# from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
#
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # ============================ CONFIGURATION =============================
# GPU_ONLY = True
# MAX_PARALLEL_IMAGES = 10  # Scale this based on GPU strength
# DYNAMIC_DET_SET = True
# SAVE_DIR_INSIDE_MEDIA = True  # If False, save in selected input folder
# FACE_MARGIN_RATIO = 0.14
# FACE_MARGIN_RATIO_PREVIEW = 0.06
# PREVIEW_BOX_CROP_RATIO = 0.85
# SHOW_PREVIEW_RECTANGLE = True
# VERBOSE_LOG = True
#
# DETECTION_SETS = [
#     (640, 640),
#     (800, 800),
#     (1024, 1024),
#     (1280, 1280),
#     (1600, 1600),
#     (2048, 2048),
# ]
#
# # ============================ PREPARE OUTPUT =============================
# now = datetime.now().strftime("Sorted faces (%Y-%m-%d %H-%M-%S)")
# BASE = os.path.join(settings.MEDIA_ROOT, "student_faces", now) if SAVE_DIR_INSIDE_MEDIA else None
# OUTPUT_DIRS = {}
#
# def create_dirs(base_folder):
#     os.makedirs(base_folder, exist_ok=True)
#     for size in DETECTION_SETS:
#         name = f"{size[0]}x{size[1]}"
#         OUTPUT_DIRS[name] = os.path.join(base_folder, name)
#         os.makedirs(OUTPUT_DIRS[name], exist_ok=True)
#     OUTPUT_DIRS["bad"] = os.path.join(base_folder, "bad")
#     OUTPUT_DIRS["previews"] = os.path.join(base_folder, "previews")
#     os.makedirs(OUTPUT_DIRS["bad"], exist_ok=True)
#     os.makedirs(OUTPUT_DIRS["previews"], exist_ok=True)
#     return os.path.join(base_folder, "sorted_face_log.csv")
#
# # ============================ ANALYSIS UTILS =============================
# def get_sharpness_score(gray_img):
#     return cv2.Laplacian(gray_img, cv2.CV_64F).var()
#
# def recommended_det_set(face_w, face_h):
#     size = max(face_w, face_h)
#     for w, h in DETECTION_SETS:
#         if size <= w:
#             return f"{w}x{h}"
#     return f"{DETECTION_SETS[-1][0]}x{DETECTION_SETS[-1][1]}"
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < 160 or h < 160:
#         hints.append("face too small")
#     if score < 0.6:
#         hints.append("very low score")
#     elif score < 0.75:
#         hints.append("low score")
#     elif score < 0.85:
#         hints.append("ok")
#     else:
#         hints.append("good")
#
#     if sharpness < 50:
#         hints.append("blurry")
#     elif sharpness < 100:
#         hints.append("soft")
#     else:
#         hints.append("sharp")
#     return "; ".join(hints)
#
# # ============================ FACE DETECTOR =============================
# face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
# face_app.prepare(ctx_id=0, det_size=(2048, 2048))  # default max size
#
# # ============================ PROCESS FUNCTION =============================
# summary_counts = {}
#
# def process_image(image_path, writer_lock, writer):
#     img = cv2.imread(image_path)
#     if img is None:
#         return
#     fname = os.path.basename(image_path)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     sharpness = round(get_sharpness_score(gray), 2)
#
#     faces = face_app.get(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
#     if not faces:
#         out_path = os.path.join(OUTPUT_DIRS["bad"], fname)
#         shutil.copy(image_path, out_path)
#         hint = "no face detected"
#         with writer_lock:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", hint])
#         return
#
#     face = faces[0]
#     x1, y1, x2, y2 = map(int, face.bbox)
#     face_crop = img[y1:y2, x1:x2]
#     w, h = x2 - x1, y2 - y1
#     score = round(face.det_score, 4)
#
#     det_set = recommended_det_set(w, h)
#     hint = generate_hint(w, h, score, sharpness)
#
#     # Save cropped
#     save_path = os.path.join(OUTPUT_DIRS[det_set], fname)
#     cv2.imwrite(save_path, face_crop)
#
#     # Save preview
#     preview_img = img.copy()
#     if SHOW_PREVIEW_RECTANGLE:
#         cv2.rectangle(preview_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
#     overlay_texts = [
#         f"Score: {score}",
#         f"Set: {det_set}",
#         f"Size: {w}x{h}",
#         f"Sharp: {sharpness}"
#     ]
#     for i, line in enumerate(overlay_texts):
#         cv2.putText(preview_img, line, (10, 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
#     preview_path = os.path.join(OUTPUT_DIRS["previews"], fname)
#     cv2.imwrite(preview_path, preview_img)
#
#     # Write CSV
#     with writer_lock:
#         writer.writerow([fname, w, h, score, sharpness, det_set, hint])
#         summary_counts[det_set] = summary_counts.get(det_set, 0) + 1
#
# # ============================ MAIN ========================================
# def main():
#     from threading import Lock
#     input_folder = os.path.join(settings.MEDIA_ROOT, "student_faces", "Original")
#     all_images = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
#     csv_path = create_dirs(BASE)
#
#     lock = Lock()
#     with open(csv_path, mode='w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#
#         with ThreadPoolExecutor(max_workers=MAX_PARALLEL_IMAGES) as executor:
#             list(tqdm(executor.map(lambda p: process_image(p, lock, writer), all_images), total=len(all_images)))
#
#         # Add summary
#         writer.writerow([])
#         summary_text = ", ".join([f"{v} {k}" for k, v in summary_counts.items()])
#         writer.writerow(["SUMMARY", summary_text])
#         print(f"📊 SUMMARY: {summary_text}")
#         print(f"✅ Sorting complete. Output saved to: {BASE}")
#
# if __name__ == "__main__":
#     main()
#








# #!/usr/bin/env python3
# import os
# import sys
# import cv2
# import csv
# import shutil
# import argparse
# import textwrap
# from tqdm import tqdm
# from datetime import datetime
# import numpy as np
# from collections import defaultdict
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # === CONFIG ===
# VERBOSE_LOGS = True
# DRAW_COLOR_RECT = True
# SAVE_IN_MEDIA = False
# FACE_MARGIN_RATIO_CROP = 0
# FACE_MARGIN_RATIO_PREVIEW = 0
# FACE_KEEP_RATIO_PREVIEW = 0
# MIN_ACCEPTED_FACE_SIZE = 160
# DETECTION_BUCKETS = [640, 800, 1024, 1280, 1600, 2048]
#
# parser = argparse.ArgumentParser(description="Sort face images by detection size recommendation")
# parser.add_argument("--input", required=True, help="Input folder containing images")
# args = parser.parse_args()
# INPUT_FOLDER = os.path.abspath(args.input)
# if not os.path.exists(INPUT_FOLDER):
#     print(f"[❌] Input folder does not exist: {INPUT_FOLDER}")
#     sys.exit(1)
#
# timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# BASE = os.path.join(settings.MEDIA_ROOT if SAVE_IN_MEDIA else os.path.dirname(INPUT_FOLDER), f"Sorted faces ({timestamp})")
# PREVIEW_DIR = os.path.join(BASE, "previews")
# LEGEND_SRC = os.path.join(settings.MEDIA_ROOT, "student_faces", "legend_reference", "legend.jpg")
# LEGEND_DEST = os.path.join(PREVIEW_DIR, "legend.jpg")
# LOG_FILE = os.path.join(BASE, "sorted_face_log.csv")
#
# REQUIRED_DIRS = ["bad", "previews"] + [f"{d}x{d}" for d in DETECTION_BUCKETS]
# for folder in REQUIRED_DIRS:
#     os.makedirs(os.path.join(BASE, folder), exist_ok=True)
#
# # Initialize face detection apps
# def get_face_app(det_size, providers):
#     app = FaceAnalysis(name='buffalo_l', providers=providers)
#     app.prepare(ctx_id=0 if "CUDA" in providers[0] else -1, det_size=det_size)
#     return app
#
# detectors = [
#     get_face_app((1024, 1024), ['CUDAExecutionProvider', 'CPUExecutionProvider']),
#     get_face_app((800, 800), ['CUDAExecutionProvider', 'CPUExecutionProvider']),
#     get_face_app((2048, 2048), ['CUDAExecutionProvider', 'CPUExecutionProvider']),
#     get_face_app((1024, 1024), ['CPUExecutionProvider']),
#     get_face_app((800, 800), ['CPUExecutionProvider']),
#     get_face_app((2048, 2048), ['CPUExecutionProvider']),
# ]
#
# def detect_faces_fallback(img):
#     for detector in detectors:
#         try:
#             faces = detector.get(img)
#             if faces:
#                 return faces
#         except Exception:
#             continue
#     return []
#
# def get_sharpness_score(gray_img):
#     return round(cv2.Laplacian(gray_img, cv2.CV_64F).var(), 2)
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < MIN_ACCEPTED_FACE_SIZE or h < MIN_ACCEPTED_FACE_SIZE:
#         hints.append("face too small; retake closer")
#     if score < 0.6:
#         hints.append("very low detection score; improve lighting")
#     elif score < 0.75:
#         hints.append("low score; try frontal face & better lighting")
#     elif score < 0.85:
#         hints.append("acceptable score; frontal & higher res helps")
#     else:
#         hints.append("good quality")
#     aspect_ratio = w / max(h, 1)
#     if aspect_ratio < 0.75 or aspect_ratio > 1.5:
#         hints.append("non-frontal or tilted face")
#     if sharpness < 50:
#         hints.append("image blurry; stabilize camera")
#     elif sharpness < 100:
#         hints.append("slightly soft image; avoid motion blur")
#     else:
#         hints.append("sharp image")
#     return "; ".join(hints)
#
# def recommend_det_set(w, h):
#     longest = max(w, h)
#     for size in DETECTION_BUCKETS:
#         if longest <= size:
#             return f"{size}x{size}"
#     return f"{DETECTION_BUCKETS[-1]}x{DETECTION_BUCKETS[-1]}"
#
# def expand_bbox(x1, y1, x2, y2, img_shape, margin_ratio):
#     h, w = img_shape[:2]
#     pad_x = int((x2 - x1) * margin_ratio)
#     pad_y = int((y2 - y1) * margin_ratio)
#     return max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
#
# def shrink_bbox(x1, y1, x2, y2, keep_ratio):
#     cx = (x1 + x2) // 2
#     cy = (y1 + y2) // 2
#     half_w = (x2 - x1) * keep_ratio / 2
#     half_h = (y2 - y1) * keep_ratio / 2
#     return int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)
#
# summary_counts = defaultdict(int)
#
# with open(LOG_FILE, mode='w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#
#     for fname in tqdm(os.listdir(INPUT_FOLDER), dynamic_ncols=True, leave=False):
#         if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#         src_path = os.path.join(INPUT_FOLDER, fname)
#         img = cv2.imread(src_path)
#         if img is None:
#             tqdm.write(f"[⚠️] Could not read {fname}")
#             continue
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         sharpness = get_sharpness_score(gray)
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faces = detect_faces_fallback(rgb)
#
#         preview_img = img.copy()
#         color = (0, 0, 255)
#         hint = "no face detected"
#         det_folder = "bad"
#         w = h = score = "-"
#
#         if faces:
#             face = faces[0]
#             score = round(face.det_score, 3)
#             x1, y1, x2, y2 = face.bbox.astype(int)
#             x1c, y1c, x2c, y2c = expand_bbox(x1, y1, x2, y2, img.shape, FACE_MARGIN_RATIO_CROP)
#             x1p, y1p, x2p, y2p = expand_bbox(x1, y1, x2, y2, img.shape, FACE_MARGIN_RATIO_PREVIEW)
#             x1p, y1p, x2p, y2p = shrink_bbox(x1p, y1p, x2p, y2p, FACE_KEEP_RATIO_PREVIEW)
#
#             face_crop = img[y1c:y2c, x1c:x2c]
#             h, w = y2 - y1, x2 - x1
#
#             if h >= MIN_ACCEPTED_FACE_SIZE and w >= MIN_ACCEPTED_FACE_SIZE:
#                 det_folder = recommend_det_set(w, h)
#                 hint = generate_hint(w, h, score, sharpness)
#                 cv2.imwrite(os.path.join(BASE, det_folder, fname), face_crop)
#             else:
#                 hint = "face too small; retake closer"
#
#             if DRAW_COLOR_RECT:
#                 color = (0, 0, 255) if det_folder == "bad" else                         (0, 255, 255) if score < 0.70 else                         (0, 165, 255) if score < 0.80 else                         (0, 255, 0) if score < 0.90 else                         (255, 0, 0)
#                 cv2.rectangle(preview_img, (x1p, y1p), (x2p, y2p), color, 2)
#
#         else:
#             shutil.copy(src_path, os.path.join(BASE, "bad", fname))
#
#         writer.writerow([fname, w, h, score, sharpness, det_folder, hint])
#         summary_counts[det_folder] += 1
#
#         font_scale = 0.75
#         y_base = img.shape[0] - 80
#         spacing = 24
#         lines = [
#             f"Score: {score}",
#             f"Set: {det_folder}",
#             f"Size: {w}x{h}",
#             f"Sharp: {sharpness}",
#         ] + textwrap.wrap(hint, width=50)
#
#         for i, line in enumerate(lines):
#             cv2.putText(preview_img, line, (10, y_base + i * spacing),
#                         cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
#
#         cv2.imwrite(os.path.join(PREVIEW_DIR, fname), preview_img)
#
# # Summary
# summary_line = ", ".join(f"{summary_counts[f]} {f}" for f in sorted(summary_counts))
# with open(LOG_FILE, "a") as f:
#     f.write("\nSUMMARY: " + summary_line + "\n")
# print("\n📊 SUMMARY: " + summary_line)
#
# if os.path.exists(LEGEND_SRC):
#     shutil.copy(LEGEND_SRC, LEGEND_DEST)
#     print(f"[ℹ️] Legend image copied to: {LEGEND_DEST}")
# print(f"✅ Sorting complete. Output saved to: {BASE}")










# #!/usr/bin/env python3
# import os
# import sys
# import cv2
# import csv
# import shutil
# import argparse
# import textwrap
# from tqdm import tqdm
# from datetime import datetime
# import numpy as np
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # === CONFIG ===
# VERBOSE_LOGS = True
# DRAW_COLOR_RECT = True
# SAVE_IN_MEDIA = False
# FACE_MARGIN_RATIO_CROP = 0.2
# FACE_MARGIN_RATIO_PREVIEW = 0.05
# FACE_KEEP_RATIO_PREVIEW = 0.85
# MIN_ACCEPTED_FACE_SIZE = 160
# DETECTION_BUCKETS = [640, 800, 1024, 1280, 1600, 2048]
#
# parser = argparse.ArgumentParser(description="Sort face images by detection size recommendation")
# parser.add_argument("--input", required=True, help="Input folder containing images")
# args = parser.parse_args()
# INPUT_FOLDER = os.path.abspath(args.input)
# if not os.path.exists(INPUT_FOLDER):
#     print(f"[❌] Input folder does not exist: {INPUT_FOLDER}")
#     sys.exit(1)
#
# timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# BASE = os.path.join(settings.MEDIA_ROOT if SAVE_IN_MEDIA else os.path.dirname(INPUT_FOLDER), f"Sorted faces ({timestamp})")
# PREVIEW_DIR = os.path.join(BASE, "previews")
# LEGEND_SRC = os.path.join(settings.MEDIA_ROOT, "student_faces", "legend_reference", "legend.jpg")
# LEGEND_DEST = os.path.join(PREVIEW_DIR, "legend.jpg")
# LOG_FILE = os.path.join(BASE, "sorted_face_log.csv")
#
# REQUIRED_DIRS = ["bad", "previews"] + [f"{d}x{d}" for d in DETECTION_BUCKETS]
# for folder in REQUIRED_DIRS:
#     os.makedirs(os.path.join(BASE, folder), exist_ok=True)
#
# # Model Initialization
# def get_face_app(det_size):
#     app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     app.prepare(ctx_id=0, det_size=det_size)
#     return app
#
# default_app = get_face_app((1024, 1024))
# fallback_app = get_face_app((800, 800))
# rescue_app = get_face_app((2048, 2048))
#
# def get_sharpness_score(gray_img):
#     return round(cv2.Laplacian(gray_img, cv2.CV_64F).var(), 2)
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#     if w < MIN_ACCEPTED_FACE_SIZE or h < MIN_ACCEPTED_FACE_SIZE:
#         hints.append("face too small; retake closer")
#     if score < 0.6:
#         hints.append("very low detection score; improve lighting")
#     elif score < 0.75:
#         hints.append("low score; try frontal face & better lighting")
#     elif score < 0.85:
#         hints.append("acceptable score; frontal & higher res helps")
#     else:
#         hints.append("good quality")
#     aspect_ratio = w / max(h, 1)
#     if aspect_ratio < 0.75 or aspect_ratio > 1.5:
#         hints.append("non-frontal or tilted face")
#     if sharpness < 50:
#         hints.append("image blurry; stabilize camera")
#     elif sharpness < 100:
#         hints.append("slightly soft image; avoid motion blur")
#     else:
#         hints.append("sharp image")
#     return "; ".join(hints)
#
# def recommend_det_set(w, h):
#     longest = max(w, h)
#     for size in DETECTION_BUCKETS:
#         if longest <= size:
#             return f"{size}x{size}"
#     return f"{DETECTION_BUCKETS[-1]}x{DETECTION_BUCKETS[-1]}"
#
# def expand_bbox(x1, y1, x2, y2, img_shape, margin_ratio):
#     h, w = img_shape[:2]
#     pad_x = int((x2 - x1) * margin_ratio)
#     pad_y = int((y2 - y1) * margin_ratio)
#     return max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
#
# def shrink_bbox(x1, y1, x2, y2, keep_ratio):
#     cx = (x1 + x2) // 2
#     cy = (y1 + y2) // 2
#     half_w = (x2 - x1) * keep_ratio / 2
#     half_h = (y2 - y1) * keep_ratio / 2
#     return int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)
#
# def detect_faces_safely(img):
#     for app in [default_app, fallback_app, rescue_app]:
#         try:
#             faces = app.get(img)
#             if faces:
#                 return faces
#         except Exception:
#             continue
#     return []
#
# with open(LOG_FILE, mode='w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#
#     for fname in tqdm(os.listdir(INPUT_FOLDER), dynamic_ncols=True, leave=False):
#         if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#         src_path = os.path.join(INPUT_FOLDER, fname)
#         img = cv2.imread(src_path)
#         if img is None:
#             tqdm.write(f"[⚠️] Could not read {fname}")
#             continue
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         sharpness = get_sharpness_score(gray)
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faces = detect_faces_safely(rgb)
#
#         preview_img = img.copy()
#         color = (0, 0, 255)
#         hint = "no face detected"
#         det_folder = "bad"
#         w = h = score = "-"
#
#         if faces:
#             face = faces[0]
#             score = round(face.det_score, 3)
#             x1, y1, x2, y2 = face.bbox.astype(int)
#             x1c, y1c, x2c, y2c = expand_bbox(x1, y1, x2, y2, img.shape, FACE_MARGIN_RATIO_CROP)
#             x1p, y1p, x2p, y2p = expand_bbox(x1, y1, x2, y2, img.shape, FACE_MARGIN_RATIO_PREVIEW)
#             x1p, y1p, x2p, y2p = shrink_bbox(x1p, y1p, x2p, y2p, FACE_KEEP_RATIO_PREVIEW)
#
#             face_crop = img[y1c:y2c, x1c:x2c]
#             h, w = y2 - y1, x2 - x1
#
#             if h >= MIN_ACCEPTED_FACE_SIZE and w >= MIN_ACCEPTED_FACE_SIZE:
#                 det_folder = recommend_det_set(w, h)
#                 hint = generate_hint(w, h, score, sharpness)
#                 cv2.imwrite(os.path.join(BASE, det_folder, fname), face_crop)
#             else:
#                 hint = "face too small; retake closer"
#
#             if DRAW_COLOR_RECT:
#                 color = (0, 0, 255) if det_folder == "bad" else                         (0, 255, 255) if score < 0.70 else                         (0, 165, 255) if score < 0.80 else                         (0, 255, 0) if score < 0.90 else                         (255, 0, 0)
#                 cv2.rectangle(preview_img, (x1p, y1p), (x2p, y2p), color, 2)
#
#         else:
#             shutil.copy(src_path, os.path.join(BASE, "bad", fname))
#
#         writer.writerow([fname, w, h, score, sharpness, det_folder, hint])
#
#         font_scale = 0.75
#         y_base = img.shape[0] - 80
#         spacing = 24
#         lines = [
#             f"Score: {score}",
#             f"Set: {det_folder}",
#             f"Size: {w}x{h}",
#             f"Sharp: {sharpness}",
#         ] + textwrap.wrap(hint, width=50)
#
#         for i, line in enumerate(lines):
#             cv2.putText(preview_img, line, (10, y_base + i * spacing),
#                         cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
#
#         cv2.imwrite(os.path.join(PREVIEW_DIR, fname), preview_img)
#
# if os.path.exists(LEGEND_SRC):
#     shutil.copy(LEGEND_SRC, LEGEND_DEST)
#     print(f"[ℹ️] Legend image copied to: {LEGEND_DEST}")
# print(f"✅ Sorting complete. Output saved to: {BASE}")












# #!/usr/bin/env python3
# import os
# import sys
# import cv2
# import csv
# import shutil
# import argparse
# import textwrap
# from tqdm import tqdm
# from datetime import datetime
# import numpy as np
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # === CONFIG ===
# VERBOSE_LOGS = True                    # Toggle terminal log output
# DRAW_COLOR_RECT = True                # Toggle colored rectangles on preview
# SAVE_IN_MEDIA = False                 # Save results in /media instead of next to input
# FACE_MARGIN_RATIO_CROP = 0         # Margin for cropped faces
# FACE_MARGIN_RATIO_PREVIEW = 0     # Margin for preview overlays
# MIN_ACCEPTED_FACE_SIZE = 160         # Min width or height in px to be considered valid
# DETECTION_BUCKETS = [640, 800, 1024, 1280, 1600, 2048]
#
# # === Parse CLI ===
# parser = argparse.ArgumentParser(description="Sort face images by detection size recommendation")
# parser.add_argument("--input", required=True, help="Input folder containing images")
# args = parser.parse_args()
# INPUT_FOLDER = os.path.abspath(args.input)
# if not os.path.exists(INPUT_FOLDER):
#     print(f"[❌] Input folder does not exist: {INPUT_FOLDER}")
#     sys.exit(1)
#
# # === Output Folders ===
# timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# BASE = os.path.join(settings.MEDIA_ROOT if SAVE_IN_MEDIA else os.path.dirname(INPUT_FOLDER), f"Sorted faces ({timestamp})")
# PREVIEW_DIR = os.path.join(BASE, "previews")
# LEGEND_SRC = os.path.join(settings.MEDIA_ROOT, "student_faces", "legend_reference", "legend.jpg")
# LEGEND_DEST = os.path.join(PREVIEW_DIR, "legend.jpg")
# LOG_FILE = os.path.join(BASE, "sorted_face_log.csv")
#
# REQUIRED_DIRS = ["bad", "previews"] + [f"{d}x{d}" for d in DETECTION_BUCKETS]
# for folder in REQUIRED_DIRS:
#     os.makedirs(os.path.join(BASE, folder), exist_ok=True)
#
# # === Model Setup ===
# face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))
#
# # === Utilities ===
# def get_sharpness_score(gray_img):
#     return round(cv2.Laplacian(gray_img, cv2.CV_64F).var(), 2)
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#
#     if w < MIN_ACCEPTED_FACE_SIZE or h < MIN_ACCEPTED_FACE_SIZE:
#         hints.append("face too small; retake closer")
#     if score < 0.6:
#         hints.append("very low detection score; improve lighting")
#     elif score < 0.75:
#         hints.append("low score; try frontal face & better lighting")
#     elif score < 0.85:
#         hints.append("acceptable score; frontal & higher res helps")
#     else:
#         hints.append("good quality")
#
#     aspect_ratio = w / max(h, 1)
#     if aspect_ratio < 0.75 or aspect_ratio > 1.5:
#         hints.append("non-frontal or tilted face")
#
#     if sharpness < 50:
#         hints.append("image blurry; stabilize camera")
#     elif sharpness < 100:
#         hints.append("slightly soft image; avoid motion blur")
#     else:
#         hints.append("sharp image")
#
#     return "; ".join(hints)
#
# def recommend_det_set(w, h):
#     longest = max(w, h)
#     for size in DETECTION_BUCKETS:
#         if longest <= size:
#             return f"{size}x{size}"
#     return f"{DETECTION_BUCKETS[-1]}x{DETECTION_BUCKETS[-1]}"
#
# def expand_bbox(x1, y1, x2, y2, img_shape, margin_ratio):
#     h, w = img_shape[:2]
#     pad_x = int((x2 - x1) * margin_ratio)
#     pad_y = int((y2 - y1) * margin_ratio)
#     return max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
#
# # === Processing Loop ===
# with open(LOG_FILE, mode='w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#
#     for fname in tqdm(os.listdir(INPUT_FOLDER), dynamic_ncols=True, leave=False):
#         if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#
#         src_path = os.path.join(INPUT_FOLDER, fname)
#         img = cv2.imread(src_path)
#         if img is None:
#             tqdm.write(f"[⚠️] Could not read {fname}")
#             continue
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         sharpness = get_sharpness_score(gray)
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         if not faces:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", "no face detected"])
#             shutil.copy(src_path, os.path.join(BASE, "bad", fname))
#             if VERBOSE_LOGS:
#                 tqdm.write(f"[🚫] {fname}: no face detected")
#             continue
#
#         face = faces[0]
#         score = round(face.det_score, 3)
#         x1, y1, x2, y2 = face.bbox.astype(int)
#         x1c, y1c, x2c, y2c = expand_bbox(x1, y1, x2, y2, img.shape, FACE_MARGIN_RATIO_CROP)
#         x1p, y1p, x2p, y2p = expand_bbox(x1, y1, x2, y2, img.shape, FACE_MARGIN_RATIO_PREVIEW)
#
#         face_crop = img[y1c:y2c, x1c:x2c]
#         h, w = y2 - y1, x2 - x1
#
#         if h < MIN_ACCEPTED_FACE_SIZE or w < MIN_ACCEPTED_FACE_SIZE:
#             det_folder = "bad"
#         else:
#             det_folder = recommend_det_set(w, h)
#
#         hint = generate_hint(w, h, score, sharpness)
#         if det_folder == "bad":
#             shutil.copy(src_path, os.path.join(BASE, "bad", fname))
#         else:
#             cv2.imwrite(os.path.join(BASE, det_folder, fname), face_crop)
#
#         writer.writerow([fname, w, h, score, sharpness, det_folder, hint])
#
#         # === Preview Image ===
#         preview_img = img.copy()
#         if DRAW_COLOR_RECT:
#             color = (0, 0, 255) if det_folder == "bad" else                     (128, 0, 128) if score < 0.60 else                     (0, 255, 255) if score < 0.70 else                     (0, 165, 255) if score < 0.80 else                     (0, 255, 0) if score < 0.90 else                     (255, 0, 0)
#
#             cv2.rectangle(preview_img, (x1p, y1p), (x2p, y2p), color, 2)
#             font_scale = 0.65
#             y_base = y2p + 20
#             spacing = 22
#             lines = [
#                 f"Score: {score:.3f}",
#                 f"Set: {det_folder}",
#                 f"Size: {w}x{h}",
#                 f"Sharp: {sharpness}",
#             ] + textwrap.wrap(hint, width=50)
#
#             for i, line in enumerate(lines):
#                 cv2.putText(preview_img, line, (10, y_base + i * spacing),
#                             cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
#
#         cv2.imwrite(os.path.join(PREVIEW_DIR, fname), preview_img)
#
# # === Legend ===
# if os.path.exists(LEGEND_SRC):
#     shutil.copy(LEGEND_SRC, LEGEND_DEST)
#     print(f"[ℹ️] Legend image copied to: {LEGEND_DEST}")
#
# print(f"✅ Sorting complete. Output saved to: {BASE}")
#
#
# # === Preview Shrink Logic ===
#
# # Add this new config under the existing preview margin config
# FACE_KEEP_RATIO_PREVIEW = 0.85     # Percentage of face bbox to retain in preview (e.g., 0.85 = keep 85%, trim 15%)
#
# # Replace preview bbox expansion logic to include cropping inward
# def shrink_bbox(x1, y1, x2, y2, keep_ratio):
#     cx = (x1 + x2) // 2
#     cy = (y1 + y2) // 2
#     half_w = (x2 - x1) * keep_ratio / 2
#     half_h = (y2 - y1) * keep_ratio / 2
#     return int(cx - half_w), int(cy - half_h), int(cx + half_w), int(cy + half_h)
#
# # Use this inside the loop when creating the preview box:
# x1p, y1p, x2p, y2p = expand_bbox(x1, y1, x2, y2, img.shape, FACE_MARGIN_RATIO_PREVIEW)
# x1p, y1p, x2p, y2p = shrink_bbox(x1p, y1p, x2p, y2p, FACE_KEEP_RATIO_PREVIEW)











# #!/usr/bin/env python3
# import os
# import sys
# import cv2
# import csv
# import shutil
# import argparse
# import textwrap
# from tqdm import tqdm
# from datetime import datetime
# import numpy as np
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # === CONFIG ===
# VERBOSE_LOGS = True                    # Toggle terminal log output
# DRAW_COLOR_RECT = True                # Toggle colored rectangles on preview
# SAVE_IN_MEDIA = False                 # Save results in /media instead of next to input
# FACE_MARGIN_RATIO_CROP = 0         # Margin for cropped faces
# FACE_MARGIN_RATIO_PREVIEW = 0.05     # Margin for preview overlays
# MIN_ACCEPTED_FACE_SIZE = 160         # Min width or height in px to be considered valid
# DETECTION_BUCKETS = [640, 800, 1024, 1280, 1600, 2048]
#
# # === Parse CLI ===
# parser = argparse.ArgumentParser(description="Sort face images by detection size recommendation")
# parser.add_argument("--input", required=True, help="Input folder containing images")
# args = parser.parse_args()
# INPUT_FOLDER = os.path.abspath(args.input)
# if not os.path.exists(INPUT_FOLDER):
#     print(f"[❌] Input folder does not exist: {INPUT_FOLDER}")
#     sys.exit(1)
#
# # === Output Folders ===
# timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# BASE = os.path.join(settings.MEDIA_ROOT if SAVE_IN_MEDIA else os.path.dirname(INPUT_FOLDER), f"Sorted faces ({timestamp})")
# PREVIEW_DIR = os.path.join(BASE, "previews")
# LEGEND_SRC = os.path.join(settings.MEDIA_ROOT, "student_faces", "legend_reference", "legend.jpg")
# LEGEND_DEST = os.path.join(PREVIEW_DIR, "legend.jpg")
# LOG_FILE = os.path.join(BASE, "sorted_face_log.csv")
#
# REQUIRED_DIRS = ["bad", "previews"] + [f"{d}x{d}" for d in DETECTION_BUCKETS]
# for folder in REQUIRED_DIRS:
#     os.makedirs(os.path.join(BASE, folder), exist_ok=True)
#
# # === Model Setup ===
# face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))
#
# # === Utilities ===
# def get_sharpness_score(gray_img):
#     return round(cv2.Laplacian(gray_img, cv2.CV_64F).var(), 2)
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#
#     if w < MIN_ACCEPTED_FACE_SIZE or h < MIN_ACCEPTED_FACE_SIZE:
#         hints.append("face too small; retake closer")
#     if score < 0.6:
#         hints.append("very low detection score; improve lighting")
#     elif score < 0.75:
#         hints.append("low score; try frontal face & better lighting")
#     elif score < 0.85:
#         hints.append("acceptable score; frontal & higher res helps")
#     else:
#         hints.append("good quality")
#
#     aspect_ratio = w / max(h, 1)
#     if aspect_ratio < 0.75 or aspect_ratio > 1.5:
#         hints.append("non-frontal or tilted face")
#
#     if sharpness < 50:
#         hints.append("image blurry; stabilize camera")
#     elif sharpness < 100:
#         hints.append("slightly soft image; avoid motion blur")
#     else:
#         hints.append("sharp image")
#
#     return "; ".join(hints)
#
# def recommend_det_set(w, h):
#     longest = max(w, h)
#     for size in DETECTION_BUCKETS:
#         if longest <= size:
#             return f"{size}x{size}"
#     return f"{DETECTION_BUCKETS[-1]}x{DETECTION_BUCKETS[-1]}"
#
# def expand_bbox(x1, y1, x2, y2, img_shape, margin_ratio):
#     h, w = img_shape[:2]
#     pad_x = int((x2 - x1) * margin_ratio)
#     pad_y = int((y2 - y1) * margin_ratio)
#     return max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
#
# # === Processing Loop ===
# with open(LOG_FILE, mode='w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#
#     for fname in tqdm(os.listdir(INPUT_FOLDER), dynamic_ncols=True, leave=False):
#         if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#
#         src_path = os.path.join(INPUT_FOLDER, fname)
#         img = cv2.imread(src_path)
#         if img is None:
#             tqdm.write(f"[⚠️] Could not read {fname}")
#             continue
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         sharpness = get_sharpness_score(gray)
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         if not faces:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", "no face detected"])
#             shutil.copy(src_path, os.path.join(BASE, "bad", fname))
#             if VERBOSE_LOGS:
#                 tqdm.write(f"[🚫] {fname}: no face detected")
#             continue
#
#         face = faces[0]
#         score = round(face.det_score, 3)
#         x1, y1, x2, y2 = face.bbox.astype(int)
#         x1c, y1c, x2c, y2c = expand_bbox(x1, y1, x2, y2, img.shape, FACE_MARGIN_RATIO_CROP)
#         x1p, y1p, x2p, y2p = expand_bbox(x1, y1, x2, y2, img.shape, FACE_MARGIN_RATIO_PREVIEW)
#
#         face_crop = img[y1c:y2c, x1c:x2c]
#         h, w = y2 - y1, x2 - x1
#
#         if h < MIN_ACCEPTED_FACE_SIZE or w < MIN_ACCEPTED_FACE_SIZE:
#             det_folder = "bad"
#         else:
#             det_folder = recommend_det_set(w, h)
#
#         hint = generate_hint(w, h, score, sharpness)
#         if det_folder == "bad":
#             shutil.copy(src_path, os.path.join(BASE, "bad", fname))
#         else:
#             cv2.imwrite(os.path.join(BASE, det_folder, fname), face_crop)
#
#         writer.writerow([fname, w, h, score, sharpness, det_folder, hint])
#
#         # === Preview Image ===
#         preview_img = img.copy()
#         if DRAW_COLOR_RECT:
#             color = (0, 0, 255) if det_folder == "bad" else                     (128, 0, 128) if score < 0.60 else                     (0, 255, 255) if score < 0.70 else                     (0, 165, 255) if score < 0.80 else                     (0, 255, 0) if score < 0.90 else                     (255, 0, 0)
#
#             cv2.rectangle(preview_img, (x1p, y1p), (x2p, y2p), color, 2)
#             font_scale = 0.65
#             y_base = y2p + 20
#             spacing = 22
#             lines = [
#                 f"Score: {score:.3f}",
#                 f"Set: {det_folder}",
#                 f"Size: {w}x{h}",
#                 f"Sharp: {sharpness}",
#             ] + textwrap.wrap(hint, width=50)
#
#             for i, line in enumerate(lines):
#                 cv2.putText(preview_img, line, (10, y_base + i * spacing),
#                             cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
#
#         cv2.imwrite(os.path.join(PREVIEW_DIR, fname), preview_img)
#
# # === Legend ===
# if os.path.exists(LEGEND_SRC):
#     shutil.copy(LEGEND_SRC, LEGEND_DEST)
#     print(f"[ℹ️] Legend image copied to: {LEGEND_DEST}")
#
# print(f"✅ Sorting complete. Output saved to: {BASE}")











# import os
# import sys
# import cv2
# import csv
# import shutil
# import argparse
# import textwrap
# from tqdm import tqdm
# from datetime import datetime
# import numpy as np
#
# # Setup Django
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "BISK_RFv3.settings")
# import django
# django.setup()
# from insightface.app import FaceAnalysis
# from django.conf import settings
#
# # === CONFIG ===
# VERBOSE_LOGS = True                   # Toggle terminal output
# DRAW_COLOR_RECT = True               # Toggle colored rectangles on preview
# FACE_MARGIN_RATIO = 0.2              # Extra margin (e.g., 0.2 means +20%)
# DETECTION_BUCKETS = [640, 800, 1024, 1280, 1600, 2048]
#
# # === Parse CLI Argument ===
# parser = argparse.ArgumentParser(description="Sort student faces by recommended detection size")
# parser.add_argument("--input", required=True, help="Folder with face images to process")
# args = parser.parse_args()
# INPUT_FOLDER = os.path.abspath(args.input)
#
# if not os.path.exists(INPUT_FOLDER):
#     print(f"[❌] Input folder does not exist: {INPUT_FOLDER}")
#     sys.exit(1)
#
# # === Output Folder Setup ===
# timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# BASE = os.path.join(os.path.dirname(INPUT_FOLDER), f"Sorted faces ({timestamp})")
# PREVIEW_DIR = os.path.join(BASE, "previews")
# LEGEND_SRC = os.path.join(settings.MEDIA_ROOT, "student_faces", "legend_reference", "legend.jpg")
# LEGEND_DEST = os.path.join(PREVIEW_DIR, "legend.jpg")
# LOG_FILE = os.path.join(BASE, "sorted_face_log.csv")
#
# REQUIRED_DIRS = ["bad", "previews"] + [f"{d}x{d}" for d in DETECTION_BUCKETS]
# for folder in REQUIRED_DIRS:
#     os.makedirs(os.path.join(BASE, folder), exist_ok=True)
#
# # === Face Model Setup ===
# face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# face_analyzer.prepare(ctx_id=0, det_size=(1024, 1024))
#
# # === Utility Functions ===
# def get_sharpness_score(gray_img):
#     return round(cv2.Laplacian(gray_img, cv2.CV_64F).var(), 2)
#
# def generate_hint(w, h, score, sharpness):
#     hints = []
#
#     if w < 160 or h < 160:
#         hints.append("face too small; retake closer")
#     if score < 0.6:
#         hints.append("very low detection score; improve lighting")
#     elif score < 0.75:
#         hints.append("low score; try frontal face & better lighting")
#     elif score < 0.85:
#         hints.append("acceptable score; frontal & higher res helps")
#     else:
#         hints.append("good quality")
#
#     aspect_ratio = w / max(h, 1)
#     if aspect_ratio < 0.75 or aspect_ratio > 1.5:
#         hints.append("non-frontal or tilted face")
#
#     if sharpness < 50:
#         hints.append("image blurry; stabilize camera")
#     elif sharpness < 100:
#         hints.append("slightly soft image; avoid motion blur")
#     else:
#         hints.append("sharp image")
#
#     return "; ".join(hints)
#
# def recommend_det_set(w, h):
#     longest = max(w, h)
#     for size in DETECTION_BUCKETS:
#         if longest <= size:
#             return f"{size}x{size}"
#     return f"{DETECTION_BUCKETS[-1]}x{DETECTION_BUCKETS[-1]}"
#
# def expand_bbox(x1, y1, x2, y2, img_shape):
#     h, w = img_shape[:2]
#     pad_x = int((x2 - x1) * FACE_MARGIN_RATIO)
#     pad_y = int((y2 - y1) * FACE_MARGIN_RATIO)
#     return max(0, x1 - pad_x), max(0, y1 - pad_y), min(w, x2 + pad_x), min(h, y2 + pad_y)
#
# # === Start Processing ===
# with open(LOG_FILE, mode='w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(["filename", "face_w", "face_h", "score", "sharpness", "recommended_det_set", "hint"])
#
#     for fname in tqdm(os.listdir(INPUT_FOLDER), dynamic_ncols=True, leave=False):
#         if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#
#         src_path = os.path.join(INPUT_FOLDER, fname)
#         img = cv2.imread(src_path)
#         if img is None:
#             if VERBOSE_LOGS:
#                 print(f"[⚠️] Could not read {fname}")
#             continue
#
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         sharpness = get_sharpness_score(gray)
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         faces = face_analyzer.get(rgb)
#
#         if not faces:
#             writer.writerow([fname, "-", "-", "-", sharpness, "bad", "no face detected"])
#             shutil.copy(src_path, os.path.join(BASE, "bad", fname))
#             if VERBOSE_LOGS:
#                 print(f"[🚫] {fname}: no face detected")
#             continue
#
#         face = faces[0]
#         score = round(face.det_score, 3)
#         x1, y1, x2, y2 = face.bbox.astype(int)
#         x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, img.shape)
#         face_crop = img[y1:y2, x1:x2]
#         h, w = y2 - y1, x2 - x1
#
#         if h < 160 or w < 160:
#             det_folder = "bad"
#         else:
#             det_folder = recommend_det_set(w, h)
#
#         hint = generate_hint(w, h, score, sharpness)
#
#         if det_folder == "bad":
#             shutil.copy(src_path, os.path.join(BASE, "bad", fname))
#         else:
#             cv2.imwrite(os.path.join(BASE, det_folder, fname), face_crop)
#
#         writer.writerow([fname, w, h, score, sharpness, det_folder, hint])
#
#         # === Generate Preview Image ===
#         preview_img = img.copy()
#         if DRAW_COLOR_RECT:
#             color = (0, 0, 255) if det_folder == "bad" else \
#                     (128, 0, 128) if score < 0.60 else \
#                     (0, 255, 255) if score < 0.70 else \
#                     (0, 165, 255) if score < 0.80 else \
#                     (0, 255, 0) if score < 0.90 else \
#                     (255, 0, 0)
#
#             cv2.rectangle(preview_img, (x1, y1), (x2, y2), color, 2)
#             cv2.putText(preview_img, f"{score:.3f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
#             cv2.putText(preview_img, f"Set: {det_folder}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#             cv2.putText(preview_img, f"Size: {w}x{h}", (x1, y2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#             cv2.putText(preview_img, f"Sharp: {sharpness}", (x1, y2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#
#             wrapped_hint = textwrap.wrap(hint, width=60)
#             for i, line in enumerate(wrapped_hint):
#                 cv2.putText(preview_img, line, (10, preview_img.shape[0] - 60 + i * 20),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
#
#         cv2.imwrite(os.path.join(PREVIEW_DIR, fname), preview_img)
#
# # === Legend Copy ===
# if os.path.exists(LEGEND_SRC):
#     shutil.copy(LEGEND_SRC, LEGEND_DEST)
#     print(f"[ℹ️] Legend image copied to: {LEGEND_DEST}")
#
# print(f"\n✅ Done. Results saved to: {BASE}")











# import os
# import cv2
# import argparse
# import shutil
# import numpy as np
# from datetime import datetime
# from insightface.app import FaceAnalysis
#
# # --- Argument Parser ---
# parser = argparse.ArgumentParser(description="Sort faces by best detection set")
# parser.add_argument("--input", required=True, help="Input folder containing images")
# args = parser.parse_args()
#
# INPUT_DIR = args.input
# if not os.path.exists(INPUT_DIR):
#     print(f"[❌] Input folder does not exist: {INPUT_DIR}")
#     exit(1)
#
# # --- Setup Detection Sets ---
# DETECTION_SIZES = ["640x640", "1024x1024", "2048x2048"]
# models = {}
#
# print(f"[🚀] Loading models for: {DETECTION_SIZES}")
# for det_size in DETECTION_SIZES:
#     w, h = map(int, det_size.split("x"))
#     model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
#     model.prepare(ctx_id=0, det_size=(w, h))
#     models[det_size] = model
#
# # --- Output Directory ---
# timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# parent_dir = os.path.dirname(os.path.abspath(INPUT_DIR))
# output_root = os.path.join(parent_dir, f"Sorted faces ({timestamp})")
# os.makedirs(output_root, exist_ok=True)
# print(f"[📁] Output folder: {output_root}")
#
# # --- Collect Images ---
# def collect_images(path):
#     image_files = []
#     for root, _, files in os.walk(path):
#         for fname in files:
#             if fname.lower().endswith((".jpg", ".jpeg", ".png")):
#                 image_files.append(os.path.join(root, fname))
#     return image_files
#
# image_paths = collect_images(INPUT_DIR)
# if not image_paths:
#     print(f"[⚠️] No images found in {INPUT_DIR}")
#     exit(0)
#
# print(f"[🔍] Found {len(image_paths)} image(s)")
#
# # --- Analyze and Sort ---
# for img_path in image_paths:
#     try:
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"[❌] Failed to load image: {img_path}")
#             continue
#
#         best_det_set = None
#         best_faces = []
#         best_score = 0
#
#         # Try all detection sets
#         for det_set, model in models.items():
#             faces = model.get(img)
#             if not faces:
#                 continue
#
#             top_face = max(faces, key=lambda f: f.det_score)
#             if top_face.det_score > best_score:
#                 best_score = top_face.det_score
#                 best_faces = [top_face]
#                 best_det_set = det_set
#
#         if not best_faces:
#             print(f"[🚫] No faces found: {os.path.basename(img_path)}")
#             continue
#
#         # Crop and save face
#         face = best_faces[0]
#         x1, y1, x2, y2 = map(int, face.bbox)
#         crop = img[y1:y2, x1:x2]
#         if crop.size == 0:
#             print(f"[⚠️] Empty crop: {img_path}")
#             continue
#
#         out_dir = os.path.join(output_root, best_det_set)
#         os.makedirs(out_dir, exist_ok=True)
#
#         out_path = os.path.join(out_dir, os.path.basename(img_path))
#         cv2.imwrite(out_path, crop)
#         print(f"[✔] {os.path.basename(img_path)} → {best_det_set} (score: {best_score:.2f})")
#
#     except Exception as e:
#         print(f"[❗] Error processing {img_path}: {e}")
#
# print(f"\n✅ Done! Check sorted images in: {output_root}")
