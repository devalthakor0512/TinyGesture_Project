"""
Step 1: Collect hand gesture data via webcam.

Uses the MediaPipe 1.0+ Tasks API (HandLandmarker) to detect 21 hand
landmark points per frame. Each frame produces 63 numbers (x, y, z per point)
which become the model's input features.

Produces: gesture_data.csv
"""

import csv
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
from hand_connections import HAND_CONNECTIONS

# ── Config ───────────────────────────────────────────────────────────────────
GESTURES           = ["open_palm", "fist", "thumbs_up"]
SAMPLES_PER_GESTURE = 200
OUTPUT_FILE        = "gesture_data.csv"
MODEL_ASSET_PATH   = "hand_landmarker.task"   # downloaded below if missing
# ─────────────────────────────────────────────────────────────────────────────


def download_model():
    """Download the MediaPipe hand landmarker model file if not present."""
    if os.path.exists(MODEL_ASSET_PATH):
        return
    print("Downloading hand_landmarker.task model (~8 MB)...")
    import urllib.request
    url = ("https://storage.googleapis.com/mediapipe-models/"
           "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
    urllib.request.urlretrieve(url, MODEL_ASSET_PATH)
    print("Download complete.\n")


def extract_landmarks(detection_result):
    """Return flattened [x,y,z]*21 list from the first detected hand, or None."""
    if not detection_result.hand_landmarks:
        return None
    hand = detection_result.hand_landmarks[0]   # first hand only
    coords = []
    for lm in hand:
        coords.extend([lm.x, lm.y, lm.z])
    return coords


def draw_landmarks(frame, detection_result):
    """Draw hand skeleton on the frame (in-place)."""
    if not detection_result.hand_landmarks:
        return
    h, w = frame.shape[:2]
    hand = detection_result.hand_landmarks[0]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 1)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def collect_gesture(gesture_name, samples_needed, writer, cap, landmarker):
    print(f"\n>>> Get ready for: {gesture_name.upper()}")
    print("    Hold your gesture in front of the camera.")
    print("    Press  S  to start recording, Q to quit.")

    recording = False
    collected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Cannot read from webcam.")
            return False

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result    = landmarker.detect(mp_image)
        draw_landmarks(frame, result)

        status = (f"Recording {gesture_name}: {collected}/{samples_needed}"
                  if recording else f"Press S to record: {gesture_name}")
        color  = (0, 255, 0) if recording else (0, 200, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("Data Collection — press Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('s'), ord('S')):
            recording = True
        if key in (ord('q'), ord('Q')):
            print("Quitting early.")
            return False

        if recording:
            coords = extract_landmarks(result)
            if coords:
                writer.writerow([gesture_name] + coords)
                collected += 1
                if collected >= samples_needed:
                    print(f"    Done! Recorded {collected} samples for {gesture_name}.")
                    break

    return True


def main():
    print("=== Stage 1: Data Collection ===")
    print(f"Gestures : {GESTURES}")
    print(f"Samples  : {SAMPLES_PER_GESTURE} per gesture")
    print(f"Output   : {OUTPUT_FILE}\n")

    download_model()

    header     = ["label"] + [f"{a}{i}" for i in range(21) for a in ("x", "y", "z")]
    file_exists = os.path.exists(OUTPUT_FILE)

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_ASSET_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        with open(OUTPUT_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(header)
            for gesture in GESTURES:
                if not collect_gesture(gesture, SAMPLES_PER_GESTURE, writer, cap, landmarker):
                    break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nData saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
