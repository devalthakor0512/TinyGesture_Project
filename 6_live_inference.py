"""
Step 6 (Optional): Live webcam demo using the quantized model.
Press Q to quit.
"""

import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, RunningMode
import torch
import torch.nn as nn
from model import GestureNet
from hand_connections import HAND_CONNECTIONS

MODEL_PATH        = "gesture_model_quant.pt"
CLASSES_FILE      = "label_encoder_classes.npy"
LANDMARKER_MODEL  = "hand_landmarker.task"


def load_gesture_model(path):
    ckpt  = torch.load(path, map_location="cpu")
    model = GestureNet(ckpt["input_dim"], ckpt["num_classes"])
    if ckpt.get("quantized"):
        model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict(model, hand_landmarks, classes):
    coords = []
    for lm in hand_landmarks:
        coords.extend([lm.x, lm.y, lm.z])
    x = torch.tensor(coords, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    idx  = probs.argmax(1).item()
    return classes[idx], probs[0, idx].item()


def draw_landmarks(frame, hand):
    h, w = frame.shape[:2]
    pts  = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
    for (a, b) in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 0), 1)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 255, 0), -1)


def main():
    print("=== Stage 6: Live Inference — press Q to quit ===")

    if not os.path.exists(LANDMARKER_MODEL):
        print("Downloading hand_landmarker.task ...")
        import urllib.request
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/"
            "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
            LANDMARKER_MODEL)

    classes = np.load(CLASSES_FILE, allow_pickle=True)
    model   = load_gesture_model(MODEL_PATH)

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=LANDMARKER_MODEL),
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
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame    = cv2.flip(frame, 1)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result   = landmarker.detect(mp_image)

            label_text = "No hand detected"
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                draw_landmarks(frame, hand)
                label, conf = predict(model, hand, classes)
                label_text  = f"{label}  ({conf*100:.1f}%)"

            cv2.putText(frame, label_text, (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 100), 2)
            cv2.imshow("Live Gesture Recognition — Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
