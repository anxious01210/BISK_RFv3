# extras/stream_utils_opencv.py
import cv2
import time


def stream_frames_opencv(rtsp_url, fps_limit=10, timeout_sec=10):
    """
    Generator that reads frames from an RTSP stream using OpenCV with FPS throttling.
    Yields RGB frames.
    """
    cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        print(f"[ERROR] Could not open camera stream: {rtsp_url}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] OpenCV camera resolution detected: {width}x{height}")

    delay = 1.0 / fps_limit
    last_frame_time = 0

    while True:
        current_time = time.time()
        if current_time - last_frame_time < delay:
            time.sleep(delay - (current_time - last_frame_time))

        ret, frame = cap.read()
        last_frame_time = time.time()

        if not ret:
            print(f"[WARNING] Failed to read frame from stream: {rtsp_url}")
            break

        yield frame

    print("[INFO] Closing OpenCV stream")
    cap.release()
