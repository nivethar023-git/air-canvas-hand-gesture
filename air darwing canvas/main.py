"""
Hand-Tracking AR User Interface
--------------------------------
Mode 1 (BASIC): green skeleton + big green circle on index fingertip (landmark 8)
Mode 2 (AR)   : futuristic cyan HUD around each palm with finger count

Keys:  m = switch mode   s = save screenshot   q = quit
"""

import math
import os
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- Settings ----------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 720
MAX_HANDS = 2
DETECT_CONF = 0.7
TRACK_CONF = 0.7

CYAN = (255, 255, 0)      # BGR
ORANGE = (0, 165, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")
HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]


# ---------------- Helpers ----------------
def to_px(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def count_fingers(pts):
    """Rotation-independent finger count using distance from wrist."""
    wrist = pts[0]
    up = 0
    # Thumb: tip farther from pinky base than the IP joint
    if dist(pts[4], pts[17]) > dist(pts[3], pts[17]) * 1.05:
        up += 1
    for tip, pip in zip(TIPS[1:], PIPS[1:]):
        if dist(pts[tip], wrist) > dist(pts[pip], wrist) * 1.1:
            up += 1
    return up


# ---------------- Mode 1: Basic ----------------
def draw_basic(frame, hands_lms):
    h, w, _ = frame.shape
    for hand in hands_lms:
        pts = [to_px(lm, w, h) for lm in hand]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], GREEN, 4)
        for p in pts:
            cv2.circle(frame, p, 6, WHITE, cv2.FILLED)
        cx, cy = pts[8]                               # index fingertip
        cv2.circle(frame, (cx, cy), 28, GREEN, cv2.FILLED)


# ---------------- Mode 2: AR HUD ----------------
def draw_hud(frame, hand, t):
    h, w, _ = frame.shape
    pts = [to_px(lm, w, h) for lm in hand]

    palm_ids = [0, 5, 9, 13, 17]
    cx = int(np.mean([pts[i][0] for i in palm_ids]))
    cy = int(np.mean([pts[i][1] for i in palm_ids]))
    center = (cx, cy)
    radius = int(dist(pts[0], pts[9]) * 1.5)
    radius = max(radius, 40)

    overlay = frame.copy()

    # Rotating arcs
    ang = int(t * 90) % 360
    cv2.ellipse(overlay, center, (radius, radius), 0, ang, ang + 100, CYAN, 4)
    cv2.ellipse(overlay, center, (radius, radius), 0, ang + 180, ang + 280, CYAN, 4)
    r2 = int(radius * 0.82)
    cv2.ellipse(overlay, center, (r2, r2), 0, -ang, -ang + 140, CYAN, 3)
    cv2.circle(overlay, center, int(radius * 1.12), CYAN, 1)

    # Tick marks
    for a in range(0, 360, 15):
        rad = math.radians(a + ang * 0.5)
        p1 = (int(cx + math.cos(rad) * radius * 1.12), int(cy + math.sin(rad) * radius * 1.12))
        p2 = (int(cx + math.cos(rad) * radius * 1.2), int(cy + math.sin(rad) * radius * 1.2))
        cv2.line(overlay, p1, p2, CYAN, 2)

    # Skeleton (thin, cyan/red like the reference)
    for a, b in HAND_CONNECTIONS:
        cv2.line(overlay, pts[a], pts[b], (80, 80, 255), 2)

    # Lines from palm center to each fingertip
    for tip in TIPS:
        cv2.line(overlay, center, pts[tip], CYAN, 1)

    # Glow blend
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Fingertip dots + joints
    for tip in TIPS:
        cv2.circle(frame, pts[tip], 11, ORANGE, cv2.FILLED)
        cv2.circle(frame, pts[tip], 14, WHITE, 1)
    for i, p in enumerate(pts):
        if i not in TIPS:
            cv2.circle(frame, p, 4, (80, 80, 255), cv2.FILLED)

    # Finger count
    n = count_fingers(pts)
    cv2.putText(frame, str(n), (cx - 15, cy + 15), cv2.FONT_HERSHEY_DUPLEX,
                1.6, WHITE, 3, cv2.LINE_AA)

    # Pinch distance (thumb tip <-> index tip)
    d = int(dist(pts[4], pts[8]))
    cv2.line(frame, pts[4], pts[8], ORANGE, 2)
    mid = ((pts[4][0] + pts[8][0]) // 2, (pts[4][1] + pts[8][1]) // 2)
    cv2.putText(frame, f"{d}px", (mid[0] + 8, mid[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2, cv2.LINE_AA)


# ---------------- Main ----------------
def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        print("Cannot open webcam. Try CAM_INDEX = 1.")
        return

    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=open(MODEL_PATH, "rb").read()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECT_CONF,
        min_hand_presence_confidence=DETECT_CONF,
        min_tracking_confidence=TRACK_CONF,
    )
    hands = vision.HandLandmarker.create_from_options(options)

    mode = "AR"
    prev = time.time()
    start = prev
    shot = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = hands.detect_for_video(mp_img, int((time.time() - start) * 1000))

        if results.hand_landmarks:
            if mode == "BASIC":
                draw_basic(frame, results.hand_landmarks)
            else:
                t = time.time() - start
                for hand in results.hand_landmarks:
                    draw_hud(frame, hand, t)

        # FPS + mode label
        now = time.time()
        fps = 1 / (now - prev) if now != prev else 0
        prev = now
        cv2.putText(frame, f"FPS: {int(fps)}  Mode: {mode}  [m] switch [q] quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, CYAN, 2, cv2.LINE_AA)

        cv2.imshow("Hand-Tracking AR UI", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            mode = "BASIC" if mode == "AR" else "AR"
        elif key == ord("s"):
            shot += 1
            cv2.imwrite(f"screenshot_{shot}.png", frame)

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
