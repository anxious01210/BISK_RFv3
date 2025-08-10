# ** Mehdi _ A very very very good one, but it lacks the auto GPU memory check and act mechanism.
import os
import cv2
import argparse
import sys
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

parser.add_argument("--det_sets", type=str, help="Comma-separated list of det_set sizes, e.g., 1024x1024,800x800")
parser.add_argument("--preview_image", action="store_true", help="Enable preview image generation")
parser.add_argument("--preview_overlay", action="store_true", help="Enable colored overlay on previews")
parser.add_argument("--preview_crop_margin", type=int, help="Margin percent for cropped previews")
parser.add_argument("--terminal_logs", action="store_true", help="Enable terminal log output")
parser.add_argument("--output_under_media", action="store_true", help="Output under media folder")
parser.add_argument("--max_gpu_percent", type=int, help="Max GPU memory usage percent")
parser.add_argument("--memory_check_interval", type=int, help="Interval for GPU memory check")

parser.add_argument("--text_color", type=str, help="Text color in format R,G,B")
parser.add_argument("--text_size", type=float, help="Font size")
parser.add_argument("--text_bold", action="store_true", help="Bold font")
parser.add_argument("--text_bg_color", type=str, help="Background color in R,G,B")
parser.add_argument("--text_bg_opacity", type=float, help="Background opacity")
parser.add_argument("--enable_text_bg", action="store_true", help="Enable text background")
parser.add_argument("--custom_font", action="store_true", help="Use custom font")

parser.add_argument("--input", type=str, required=True, help="Input folder of student images")
args = parser.parse_args()

if args.det_sets:
    DETECTION_SIZES = args.det_sets.split(",")

# if args.preview_image:
#     ENABLE_PREVIEW_IMAGE = True
ENABLE_PREVIEW_IMAGE = getattr(args, 'preview_image', False)
# if args.preview_overlay:
#     ENABLE_PREVIEW_OVERLAY = True
ENABLE_PREVIEW_OVERLAY = getattr(args, 'preview_overlay', False)
if args.preview_crop_margin is not None:
    PREVIEW_CROP_MARGIN_PERCENT = args.preview_crop_margin
if args.terminal_logs:
    ENABLE_TERMINAL_LOGS = True
# if args.output_under_media:
#     OUTPUT_UNDER_MEDIA_FOLDER = True
OUTPUT_UNDER_MEDIA_FOLDER = getattr(args, 'output_under_media', False)
if args.max_gpu_percent:
    MAX_GPU_MEMORY_PERCENT = args.max_gpu_percent
if args.memory_check_interval:
    MEMORY_CHECK_INTERVAL = args.memory_check_interval

if args.text_color:
    PREVIEW_TEXT_COLOR = tuple(map(int, args.text_color.split(",")))
if args.text_size:
    PREVIEW_TEXT_SIZE = args.text_size
# if args.text_bold:
    # PREVIEW_TEXT_BOLD = True
PREVIEW_TEXT_BOLD = getattr(args, 'text_bold', False)
if args.text_bg_color:
    PREVIEW_TEXT_BG_COLOR = tuple(map(int, args.text_bg_color.split(",")))
if args.text_bg_opacity is not None:
    PREVIEW_TEXT_BG_OPACITY = args.text_bg_opacity
# if args.enable_text_bg:
#     ENABLE_TEXT_BG = True
ENABLE_TEXT_BG = getattr(args, 'enable_text_bg', False)
# if args.custom_font:
#     ENABLE_CUSTOM_FONT = True
ENABLE_CUSTOM_FONT = getattr(args, 'custom_font', False)

print(f"🟨 FINAL TEXT CONFIG — COLOR: {PREVIEW_TEXT_COLOR}, BG: {PREVIEW_TEXT_BG_COLOR}, BOLD: {PREVIEW_TEXT_BOLD}, SIZE: {PREVIEW_TEXT_SIZE}")

INPUT_FOLDER = args.input
TIMESTAMP = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
# OUTPUT_BASE = os.path.join("media", f"Sorted faces ({TIMESTAMP})") if OUTPUT_UNDER_MEDIA_FOLDER else os.path.join(INPUT_FOLDER, f"Sorted faces ({TIMESTAMP})")
OUTPUT_PARENT = "media" if OUTPUT_UNDER_MEDIA_FOLDER else os.path.dirname(INPUT_FOLDER)
OUTPUT_BASE = os.path.join(OUTPUT_PARENT, f"Sorted faces ({TIMESTAMP})")
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

if os.path.isdir(INPUT_FOLDER):
    image_files = sorted([
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
else:
    image_files = [INPUT_FOLDER]
image_filenames = [os.path.basename(f) for f in image_files]
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
        # if ENABLE_TEXT_BG:
        #     overlay = preview.copy()
        #     cv2.rectangle(overlay, (x, y - th - 4), (x + tw + 2, y + 2), PREVIEW_TEXT_BG_COLOR, -1)
        #     cv2.addWeighted(overlay, PREVIEW_TEXT_BG_OPACITY, preview, 1 - PREVIEW_TEXT_BG_OPACITY, 0, preview)
        # cv2.putText(preview, text, (x, y), font, font_scale, PREVIEW_TEXT_COLOR, font_thickness)
        bg_color_bgr = tuple(reversed(PREVIEW_TEXT_BG_COLOR))
        text_color_bgr = tuple(reversed(PREVIEW_TEXT_COLOR))

        if ENABLE_TEXT_BG:
            overlay = preview.copy()
            cv2.rectangle(overlay, (x, y - th - 4), (x + tw + 2, y + 2), bg_color_bgr, -1)
            cv2.addWeighted(overlay, PREVIEW_TEXT_BG_OPACITY, preview, 1 - PREVIEW_TEXT_BG_OPACITY, 0, preview)

        cv2.putText(preview, text, (x, y), font, font_scale, text_color_bgr, font_thickness)

    return preview

print("🟢 Script started...", flush=True)
any_detector_used = False

with open(log_path, "w") as log:
    log.write("filename,det_set,result,score,bbox_width,bbox_height,x_center,y_center,blur_score,brightness,shake_score,hint\n")

    for idx, img_path in enumerate(image_files):
        fname = image_filenames[idx]
        print(f"🔄 Sorting faces... {idx + 1}/{len(image_files)} ({(idx + 1) * 100 // len(image_files)}%)", flush=True)
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

print("\n📊 Summary Report:")
for k, v in summary.items():
    print(f"    → {k:>24}: {v}")  # or "▪", "→", ">>", etc. # uses tab instead of spaces

print("✅ Sorting loop completed, preparing summary.")
print(f"📁 Output: {OUTPUT_BASE}")
print("✅ Sorting complete.", flush=True)
print("✅ Script completed.", flush=True)

# ✅ Add this after the above
if "job_id" in os.environ:
    job_log_path = os.path.join("media", "logs", "sorted_faces", f"{os.environ['job_id']}.log")
    os.makedirs(os.path.dirname(job_log_path), exist_ok=True)
    with open(job_log_path, "a") as f:
        f.write("✅ Script completed.\n")
        f.write(f"📁 Output: {OUTPUT_BASE}\n")