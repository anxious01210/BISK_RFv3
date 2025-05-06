# extras/stream_utils_ffmpeg.py
import ffmpeg
import numpy as np
import cv2
import subprocess
import os
import select

def stream_frames_ffmpeg(rtsp_url, fps_limit=10, timeout_sec=10):
    width, height = 1280, 720  # default fallback resolution

    # Try probing actual resolution
    try:
        probe = ffmpeg.probe(rtsp_url)
        video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
        if video_streams:
            width = int(video_streams[0]['width'])
            height = int(video_streams[0]['height'])
            print(f"[INFO] Camera resolution detected: {width}x{height}")
    except Exception as e:
        print(f"[WARNING] Failed to probe resolution: {e} — using fallback {width}x{height}")

    frame_size = width * height * 3
    print(f"[DEBUG] Expected frame size: {frame_size} bytes")

    # Start FFmpeg process
    process = (
        ffmpeg
        .input(rtsp_url, rtsp_transport='tcp', loglevel='quiet')
        .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=fps_limit)
        .run_async(pipe_stdout=True, pipe_stderr=True)
    )

    while True:
        # Use select to wait for data with a timeout
        rlist, _, _ = select.select([process.stdout], [], [], timeout_sec)
        if not rlist:
            print(f"[ERROR] Timeout: no frame received in {timeout_sec} seconds from {rtsp_url}")
            break

        in_bytes = process.stdout.read(frame_size)
        if not in_bytes or len(in_bytes) != frame_size:
            print(f"[ERROR] Incomplete frame or stream ended. Bytes read: {len(in_bytes) if in_bytes else 0}")
            break

        try:
            frame = np.frombuffer(in_bytes, np.uint8).reshape([height, width, 3])
            yield frame
        except Exception as e:
            print(f"[ERROR] Failed to reshape frame: {e}")
            break

    print("[INFO] Closing FFmpeg stream")
    process.terminate()
    process.wait()




# # extras/stream_utils_ffmpeg.py
# import ffmpeg
# import numpy as np
# import select
#
# def stream_frames_ffmpeg(rtsp_url, fps_limit=1, timeout_sec=10):
#     """
#     Stream frames from an RTSP URL using FFmpeg with dynamic resolution detection.
#     Falls back to 1280x720 if resolution probing fails.
#
#     Yields:
#         np.ndarray: BGR image frame.
#     """
#     width, height = 1280, 720  # Default resolution
#
#     try:
#         probe = ffmpeg.probe(rtsp_url)
#         video_streams = [s for s in probe['streams'] if s['codec_type'] == 'video']
#         if video_streams:
#             width = int(video_streams[0]['width'])
#             height = int(video_streams[0]['height'])
#             print(f"[INFO] Detected resolution: {width}x{height}")
#     except Exception as e:
#         print(f"[WARNING] FFmpeg probe failed: {e} — using fallback resolution {width}x{height}")
#
#     frame_size = width * height * 3
#     print(f"[DEBUG] Expected frame size: {frame_size} bytes")
#
#     try:
#         process = (
#             ffmpeg
#             .input(rtsp_url, rtsp_transport='tcp', loglevel='quiet')
#             .output('pipe:', format='rawvideo', pix_fmt='bgr24', r=fps_limit)
#             .run_async(pipe_stdout=True, pipe_stderr=True)
#         )
#     except Exception as e:
#         print(f"[ERROR] Failed to start FFmpeg: {e}")
#         return
#
#     while True:
#         rlist, _, _ = select.select([process.stdout], [], [], timeout_sec)
#         if not rlist:
#             print(f"[ERROR] Timeout: no frame in {timeout_sec} sec")
#             break
#
#         in_bytes = process.stdout.read(frame_size)
#         if not in_bytes or len(in_bytes) != frame_size:
#             print(f"[ERROR] Incomplete frame. Bytes read: {len(in_bytes) if in_bytes else 0}")
#             break
#
#         try:
#             frame = np.frombuffer(in_bytes, np.uint8).reshape((height, width, 3))
#             yield frame
#         except Exception as e:
#             print(f"[ERROR] Frame reshape failed: {e}")
#             break
#
#     process.terminate()
#     process.wait()
#     print("[INFO] FFmpeg stream closed")
