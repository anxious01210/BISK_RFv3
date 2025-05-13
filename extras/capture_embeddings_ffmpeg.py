import os
import cv2
import ffmpeg
import pickle
import numpy as np
from datetime import datetime
from insightface.app import FaceAnalysis

# --- Configuration ---
H_CODE = "H123456"  # ← Set this before running for each student
# CAMERA_SOURCE = 0  # 0 = webcam | "rtsp://..." = IP camera
CAMERA_SOURCE = "rtsp://admin:B!sk2025@192.168.137.95:554/Streaming/Channels/101/"  # 0 = webcam | "rtsp://..." = IP camera

SAVE_DIR = "media"
FACE_DIR = os.path.join(SAVE_DIR, "stream_faces", H_CODE)
EMBEDDING_DIR = os.path.join(SAVE_DIR, "embeddings")
PKL_PATH = os.path.join(SAVE_DIR, "face_embeddings.pkl")

DETECTION_SIZE = (2048, 2048)
MAX_FRAMES = 100
MIN_CONFIDENCE = 0.88
# MIN_CONFIDENCE = 0.80

# Ensure directories exist
os.makedirs(FACE_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# --- Load model ---
face_app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_app.prepare(ctx_id=0, det_size=DETECTION_SIZE)

# --- FFmpeg Input Setup ---
def get_ffmpeg_input(src):
    if str(src).isdigit():
        return ffmpeg.input(f"/dev/video{src}", format="v4l2", framerate=10)  # If your webcam supports 30 FPS, modify this
    elif str(src).startswith("rtsp://") or str(src).startswith("http"):
        return ffmpeg.input(src, rtsp_transport="tcp", timeout=5000000)
    else:
        raise ValueError("Unsupported camera source")

# --- Stream and Process Frames ---
def extract_frames_ffmpeg(src, max_frames=50):
    input_stream = get_ffmpeg_input(src).output('pipe:', format='rawvideo', pix_fmt='bgr24').run_async(pipe_stdout=True)
    probe = ffmpeg.probe(src) if isinstance(src, str) and src.startswith("rtsp") else {}
    width = 1280
    height = 720

    try:
        for stream in probe.get("streams", []):
            if stream["codec_type"] == "video":
                width = int(stream["width"])
                height = int(stream["height"])
                break
    except:
        print("[WARN] Failed to probe camera resolution. Using default.")

    print(f"[INFO] Streaming with resolution: {width}x{height}")
    frame_size = width * height * 3

    frames = []
    for i in range(max_frames):
        in_bytes = input_stream.stdout.read(frame_size)
        if len(in_bytes) != frame_size:
            break
        frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
        frames.append(frame)
    input_stream.stdout.close()
    input_stream.wait()
    return frames

# --- Main Logic ---
def capture_and_save_embeddings():
    print(f"[INFO] Starting capture for student {H_CODE}")
    embeddings = []
    frames = extract_frames_ffmpeg(CAMERA_SOURCE, MAX_FRAMES)

    for idx, frame in enumerate(frames):
        faces = face_app.get(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for face in faces:
            print(f"[DEBUG] Frame {idx}: Detected face with score {face.det_score:.3f}")
            if face.det_score < MIN_CONFIDENCE or face.embedding is None:
                continue

            emb = np.asarray(face.embedding, dtype=np.float32)
            embeddings.append(emb)

            # Save cropped face
            x1, y1, x2, y2 = map(int, face.bbox)
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                crop_path = os.path.join(FACE_DIR, f"face_{idx:03d}.jpg")
                cv2.imwrite(crop_path, face_crop)
                print(f"[✔] Frame {idx} | Det Score: {face.det_score:.2f} | Saved: {crop_path}")
            break  # Only first face per frame

    if not embeddings:
        print("[❌] No high-confidence faces captured.")
        return

    # --- Save .npy ---
    avg_embedding = np.mean(embeddings, axis=0)
    npy_path = os.path.join(EMBEDDING_DIR, f"{H_CODE}.npy")
    np.save(npy_path, avg_embedding)
    print(f"[✅] Saved embedding to {npy_path}")

    # --- Update .pkl ---
    face_dict = {}
    if os.path.exists(PKL_PATH):
        with open(PKL_PATH, "rb") as f:
            face_dict = pickle.load(f)

    face_dict[H_CODE] = avg_embedding
    with open(PKL_PATH, "wb") as f:
        pickle.dump(face_dict, f)
    print(f"[🆗] Updated {PKL_PATH} with {H_CODE}")

if __name__ == "__main__":
    capture_and_save_embeddings()
