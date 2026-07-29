"""
Face tracking + expression/emotion detection + hand tracking.

Builds on expression_tracking.py by adding:
  - Hand landmark tracking (up to 2 hands) via MediaPipe Hands
  - A simple rule-based "predicted emotion" label derived from the
    blendshape scores (Happy, Sad, Angry, Surprised, Disgusted, Neutral)
  - Face bounding box color that changes based on the predicted emotion
  - All on-screen text drawn on a dark, semi-transparent panel for
    readability instead of directly over the video

Note: the emotion label is a heuristic built from facial-muscle scores
(smile, brow raise, jaw open, etc.), not a trained emotion classifier -
treat it as a fun approximation, not a clinical/scientific measure.
"""

import os
import time
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

FACE_MODEL_PATH = "face_landmarker.task"
FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Standard 21-point hand skeleton connections (thumb, index, middle, ring,
# pinky, and the palm base) - used to draw hand landmarks without relying
# on the legacy mp.solutions.drawing_utils / mp.solutions.hands API.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17),                                # palm base
]

TOP_N_EXPRESSIONS = 3
EXPRESSION_THRESHOLD = 0.3
SMOOTHING_ALPHA = 0.4
BLENDSHAPE_SMOOTHING_ALPHA = 0.4

# How many frames to keep showing the last known face box/emotion after
# detection briefly drops out (blink, quick turn, motion blur) before
# actually reporting "no face detected".
FACE_GRACE_FRAMES = 12

# How many consecutive frames a new emotion must "win" before it replaces
# the currently displayed one - prevents rapid flicker between labels.
EMOTION_DEBOUNCE_FRAMES = 5

# BGR colors for each predicted emotion (used for the face box + label)
EMOTION_COLORS = {
    "Happy": (0, 220, 0),
    "Sad": (255, 120, 0),
    "Angry": (0, 0, 255),
    "Surprised": (0, 255, 255),
    "Disgusted": (200, 0, 200),
    "Neutral": (200, 200, 200),
}


def ensure_model_downloaded(path, url, label):
    if not os.path.exists(path):
        print(f"Downloading {label} model (first run only)...")
        urllib.request.urlretrieve(url, path)
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


def build_face_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
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


def build_hand_landmarker():
    base_options = mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def draw_hand_landmarks(img, hand_landmarks_list, w, h):
    for hand_landmarks in hand_landmarks_list:
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
        for start_idx, end_idx in HAND_CONNECTIONS:
            cv2.line(img, points[start_idx], points[end_idx], (255, 255, 255), 2)
        for point in points:
            cv2.circle(img, point, 4, (0, 140, 255), cv2.FILLED)


def draw_face_box(img, landmarks, w, h, color):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return (x1 + x2) // 2, (y1 + y2) // 2


def predict_emotion(scores):
    """
    Very simple rule-based mapping from blendshape scores to an emotion
    label. Not a trained classifier - just combines a few of the most
    relevant facial-muscle scores per emotion.
    """
    smile = max(scores.get("mouthSmileLeft", 0), scores.get("mouthSmileRight", 0))
    frown = max(scores.get("mouthFrownLeft", 0), scores.get("mouthFrownRight", 0))
    brow_up = max(scores.get("browInnerUp", 0),
                  scores.get("browOuterUpLeft", 0),
                  scores.get("browOuterUpRight", 0))
    brow_down = max(scores.get("browDownLeft", 0), scores.get("browDownRight", 0))
    jaw_open = scores.get("jawOpen", 0)
    eye_wide = max(scores.get("eyeWideLeft", 0), scores.get("eyeWideRight", 0))
    nose_sneer = max(scores.get("noseSneerLeft", 0), scores.get("noseSneerRight", 0))
    squint = max(scores.get("eyeSquintLeft", 0), scores.get("eyeSquintRight", 0))

    if jaw_open > 0.5 and (brow_up > 0.3 or eye_wide > 0.3):
        return "Surprised"
    if brow_down > 0.4 and nose_sneer > 0.2:
        return "Angry"
    if squint > 0.4 and nose_sneer > 0.3:
        return "Disgusted"
    if frown > 0.4 and brow_down > 0.2:
        return "Sad"
    if smile > 0.5 and jaw_open < 0.3:
        return "Happy"
    return "Neutral"


class EmotionStabilizer:
    """
    Smooths blendshape scores frame-to-frame with an EMA, and debounces
    the final emotion label so it only changes after a new prediction
    wins for several consecutive frames - avoids flicker between labels
    even when the underlying scores are noisy.
    """

    def __init__(self, alpha=BLENDSHAPE_SMOOTHING_ALPHA, debounce_frames=EMOTION_DEBOUNCE_FRAMES):
        self.alpha = alpha
        self.debounce_frames = debounce_frames
        self.smoothed_scores = {}
        self.displayed_emotion = "Neutral"
        self.candidate_emotion = "Neutral"
        self.candidate_streak = 0

    def update(self, raw_scores):
        for name, value in raw_scores.items():
            prev = self.smoothed_scores.get(name, value)
            self.smoothed_scores[name] = self.alpha * value + (1 - self.alpha) * prev

        predicted = predict_emotion(self.smoothed_scores)

        if predicted == self.candidate_emotion:
            self.candidate_streak += 1
        else:
            self.candidate_emotion = predicted
            self.candidate_streak = 1

        if self.candidate_streak >= self.debounce_frames:
            self.displayed_emotion = self.candidate_emotion

        return self.displayed_emotion


class FaceTrackState:
    """
    Holds the last known good face detection so a brief miss (blink,
    quick head turn, motion blur) doesn't make the box/label vanish for
    a single frame - it just reuses the last result until grace_frames
    of consecutive misses is exceeded.
    """

    def __init__(self, grace_frames=FACE_GRACE_FRAMES):
        self.grace_frames = grace_frames
        self.missed_frames = 0
        self.last_landmarks = None
        self.last_scores = {}
        self.has_face = False

    def update(self, face_landmarks, blendshape_scores):
        if face_landmarks is not None:
            self.last_landmarks = face_landmarks
            self.last_scores = blendshape_scores
            self.missed_frames = 0
            self.has_face = True
        else:
            self.missed_frames += 1
            if self.missed_frames > self.grace_frames:
                self.has_face = False

        return self.has_face, self.last_landmarks, self.last_scores


def draw_panel(img, lines, origin=(10, 10), line_height=26, padding=10, alpha=0.6):
    """
    Draws a dark, semi-transparent panel behind a list of (text, color)
    tuples, stacked top to bottom - keeps text readable over any
    background instead of overlaying it directly on the video.
    """
    x, y = origin
    text_sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0] for t, _ in lines]
    panel_w = max(w for w, h in text_sizes) + padding * 2
    panel_h = line_height * len(lines) + padding * 2

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + panel_w, y + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    text_y = y + padding + 18
    for text, color in lines:
        cv2.putText(img, text, (x + padding, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        text_y += line_height


def main():
    ensure_model_downloaded(FACE_MODEL_PATH, FACE_MODEL_URL, "face landmarker")
    ensure_model_downloaded(HAND_MODEL_PATH, HAND_MODEL_URL, "hand landmarker")
    cap = open_camera()
    face_landmarker = build_face_landmarker()
    hand_landmarker = build_hand_landmarker()

    stabilizer = EmotionStabilizer()
    face_state = FaceTrackState()

    smoothed_dx, smoothed_dy = 0, 0
    start_time = time.time()
    fps = 0.0
    prev_frame_time = start_time

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from camera.")
            break

        now = time.time()
        frame_dt = now - prev_frame_time
        prev_frame_time = now
        if frame_dt > 0:
            # Smooth the FPS reading itself so it doesn't jump around
            instant_fps = 1.0 / frame_dt
            fps = fps * 0.9 + instant_fps * 0.1 if fps else instant_fps

        # Real elapsed time since start, in ms - matches actual frame
        # timing instead of assuming a fixed frame rate. MediaPipe's
        # VIDEO mode uses these timestamps to reason about motion
        # between frames, so accurate timing matters for tracking quality.
        timestamp_ms = int((now - start_time) * 1000)

        h, w, _ = img.shape
        frame_center = (w // 2, h // 2)
        cv2.drawMarker(img, frame_center, (0, 255, 255), cv2.MARKER_CROSS, 20, 2)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # --- Hand tracking ---
        hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        if hand_result.hand_landmarks:
            draw_hand_landmarks(img, hand_result.hand_landmarks, w, h)

        # --- Face + expression tracking ---
        result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

        raw_landmarks = result.face_landmarks[0] if result.face_landmarks else None
        raw_scores = {}
        if result.face_blendshapes:
            raw_scores = {s.category_name: s.score for s in result.face_blendshapes[0]}

        has_face, landmarks, held_scores = face_state.update(raw_landmarks, raw_scores)

        panel_lines = [(f"FPS: {fps:.0f}", (150, 150, 150))]

        if has_face and landmarks is not None:
            emotion = stabilizer.update(held_scores)
            color = EMOTION_COLORS[emotion]

            center = draw_face_box(img, landmarks, w, h, color)
            cv2.circle(img, center, 5, color, cv2.FILLED)

            raw_dx = center[0] - frame_center[0]
            raw_dy = center[1] - frame_center[1]
            smoothed_dx = int(SMOOTHING_ALPHA * raw_dx + (1 - SMOOTHING_ALPHA) * smoothed_dx)
            smoothed_dy = int(SMOOTHING_ALPHA * raw_dy + (1 - SMOOTHING_ALPHA) * smoothed_dy)

            top_shapes = sorted(stabilizer.smoothed_scores.items(),
                                 key=lambda kv: kv[1], reverse=True)
            top_shapes = [(name, score) for name, score in top_shapes
                          if score >= EXPRESSION_THRESHOLD][:TOP_N_EXPRESSIONS]

            panel_lines.append((f"Emotion: {emotion}", color))
            panel_lines.append((f"dx={smoothed_dx} dy={smoothed_dy}", (255, 255, 255)))
            for name, score in top_shapes:
                panel_lines.append((f"{name}: {score:.2f}", (0, 255, 0)))
        else:
            panel_lines.append(("No face detected", (0, 0, 255)))

        draw_panel(img, panel_lines)

        cv2.imshow("Face + Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()