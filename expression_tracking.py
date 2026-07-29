"""
Face tracking + expression detection - computer-only version.

Upgrades over tracking.py:
  - Higher capture resolution for better detection accuracy
  - Raised confidence thresholds to reduce false positives
  - Smoothed (dx, dy) offset so tracking looks stable instead of jittery
  - Real expression detection using MediaPipe's Face Landmarker task,
    which outputs 52 named "blendshape" scores per frame (smile, brow
    raise, mouth open, eye closed, etc.) - much more reliable than
    hand-rolled landmark math.

The Face Landmarker model file is downloaded automatically the first
time you run this (about 4 MB), then cached locally as face_landmarker.task.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "face_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

# How many top expressions to show on screen each frame
TOP_N_EXPRESSIONS = 3
# Ignore blendshapes below this score (0-1) - filters out noise
EXPRESSION_THRESHOLD = 0.3
# Smoothing factor for dx/dy - lower = smoother but more lag
SMOOTHING_ALPHA = 0.4


def ensure_model_downloaded():
    if not os.path.exists(MODEL_PATH):
        print("Downloading face landmarker model (first run only)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Done.")


def open_camera(width=1280, height=720):
    for index in (0, 1, 2):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            print(f"Using camera index {index}")
            return cap
        cap.release()
    raise RuntimeError("No camera found. Try a different index or check connections.")


def build_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def draw_face_box(img, landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return (x1 + x2) // 2, (y1 + y2) // 2


def main():
    ensure_model_downloaded()
    cap = open_camera()
    landmarker = build_landmarker()

    smoothed_dx, smoothed_dy = 0, 0
    frame_index = 0

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        h, w, _ = img.shape
        frame_center = (w // 2, h // 2)
        cv2.drawMarker(img, frame_center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = int(frame_index * (1000 / 30))  # assume ~30 fps
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        frame_index += 1

        if result.face_landmarks:
            landmarks = result.face_landmarks[0]
            center = draw_face_box(img, landmarks, w, h)
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

            # Smoothed offset from center (same numbers tracking.py exposes)
            raw_dx = center[0] - frame_center[0]
            raw_dy = center[1] - frame_center[1]
            smoothed_dx = int(SMOOTHING_ALPHA * raw_dx + (1 - SMOOTHING_ALPHA) * smoothed_dx)
            smoothed_dy = int(SMOOTHING_ALPHA * raw_dy + (1 - SMOOTHING_ALPHA) * smoothed_dy)
            cv2.putText(img, f"dx={smoothed_dx} dy={smoothed_dy}", (10, 30),
                        cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)

            # Expression detection via blendshapes
            if result.face_blendshapes:
                shapes = result.face_blendshapes[0]
                top = sorted(shapes, key=lambda s: s.score, reverse=True)
                top = [s for s in top if s.score >= EXPRESSION_THRESHOLD][:TOP_N_EXPRESSIONS]

                y = 60
                for shape in top:
                    label = f"{shape.category_name}: {shape.score:.2f}"
                    cv2.putText(img, label, (10, y), cv2.FONT_HERSHEY_PLAIN,
                                1.3, (0, 255, 0), 2)
                    y += 25

        cv2.imshow("Face Tracking + Expressions", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()