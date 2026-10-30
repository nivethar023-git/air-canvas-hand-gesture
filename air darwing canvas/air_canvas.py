"""
Air Drawing Canvas
------------------
Draw in the air with your index finger.

Gestures (one hand):
  ☝  Index finger only          -> DRAW
  ✌  Index + middle fingers     -> SELECT (hover over a color / eraser at the top bar)
  🖐  Open palm (4 fingers)      -> ERASE (big eraser at your index fingertip)
  ✊  Fist (hold ~1 second)      -> CLEAR canvas

Keys: + / -  brush size    s  save drawing    c  clear    q  quit
"""

import math
import os
import sys
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# ---------------- Settings ----------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 540
BRUSH = 8
ERASER = 50
SMOOTH = 0.3            # 0 = no smoothing, closer to 1 = very smooth (laggy)
CLEAR_HOLD_SECONDS = 1.0
BAR_H = 90

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task")

# name, BGR color
PALETTE = [
    ("Red", (0, 0, 255)),
    ("Green", (0, 220, 0)),
    ("Blue", (255, 100, 0)),
    ("Yellow", (0, 230, 255)),
    ("Pink", (200, 80, 255)),
    ("White", (255, 255, 255)),
    ("Eraser", (0, 0, 0)),
]

HAND_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]


# ---------------- Helpers ----------------
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand model (one time)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def fingers_up(pts):
    """Returns [thumb, index, middle, ring, pinky] as booleans."""
    wrist = pts[0]
    state = [dist(pts[4], pts[17]) > dist(pts[3], pts[17]) * 1.05]
    for tip, pip in zip(TIPS[1:], PIPS[1:]):
        state.append(dist(pts[tip], wrist) > dist(pts[pip], wrist) * 1.1)
    return state


def open_camera():
    """Try several backends/indexes and return the first camera giving a real (non-black) image."""
    if sys.platform.startswith("win"):
        backends = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]
    else:
        backends = [("ANY", cv2.CAP_ANY)]
    indexes = [CAM_INDEX] + [i for i in range(3) if i != CAM_INDEX]

    for index in indexes:
        for name, be in backends:
            cap = cv2.VideoCapture(index, be)
            if not cap.isOpened():
                cap.release()
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)   # set size BEFORE reading frames
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)          # always use the newest frame (less lag)
            try:
                for _ in range(40):                  # warm-up: first frames are often black
                    ok, f = cap.read()
                    if ok and f is not None and f.mean() > 8:
                        print(f"Camera OK: index={index}, backend={name}")
                        return cap
                    time.sleep(0.05)
            except cv2.error:
                pass                                 # this backend is broken, try the next
            cap.release()
    return None


def draw_palette(frame, selected, hover=None):
    w = frame.shape[1]
    box_w = w // len(PALETTE)
    cv2.rectangle(frame, (0, 0), (w, BAR_H), (30, 30, 30), cv2.FILLED)
    for i, (name, color) in enumerate(PALETTE):
        x1, x2 = i * box_w, (i + 1) * box_w
        cv2.rectangle(frame, (x1 + 8, 10), (x2 - 8, BAR_H - 10), color, cv2.FILLED)
        txt_color = (0, 0, 0) if name != "Eraser" else (255, 255, 255)
        if name == "Eraser":
            cv2.rectangle(frame, (x1 + 8, 10), (x2 - 8, BAR_H - 10), (150, 150, 150), 2)
        cv2.putText(frame, name, (x1 + 18, BAR_H // 2 + 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, txt_color, 2, cv2.LINE_AA)
        if i == selected:
            cv2.rectangle(frame, (x1 + 4, 6), (x2 - 4, BAR_H - 6), (255, 255, 255), 3)
        if i == hover:
            cv2.rectangle(frame, (x1 + 4, 6), (x2 - 4, BAR_H - 6), (0, 255, 255), 2)


def merge_canvas(frame, canvas):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY_INV)
    base = cv2.bitwise_and(frame, frame, mask=inv)
    return cv2.bitwise_or(base, canvas)


# ---------------- Main ----------------
def main():
    cap = open_camera()
    if cap is None:
        print("No working camera image found.")
        print("- Close Zoom/Teams/Camera app/browser tabs using the webcam")
        print("- Windows Settings > Privacy & security > Camera: allow desktop apps")
        print("- Check the physical privacy shutter / camera key on your laptop")
        return
    ensure_model()
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=open(MODEL_PATH, "rb").read()),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    hands = vision.HandLandmarker.create_from_options(options)

    canvas = None
    selected = 0
    brush = BRUSH
    prev = None            # previous smoothed fingertip
    smooth_pt = None
    fist_start = None
    start = time.time()
    saved = 0
    last_t = time.time()
    fps = 0.0

    while True:
        try:
            ok, frame = cap.read()
        except cv2.error:
            continue
        if not ok or frame is None:
            break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        if frame.mean() < 3:
            cv2.putText(frame, "Camera image is black - check camera access", (20, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        if canvas is None or canvas.shape[:2] != (h, w):
            canvas = np.zeros((h, w, 3), np.uint8)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = hands.detect_for_video(mp_img, int((time.time() - start) * 1000))

        mode = "NO HAND"
        hover = None
        cursor = None

        if res.hand_landmarks:
            lm = res.hand_landmarks[0]
            pts = [(int(p.x * w), int(p.y * h)) for p in lm]

            # faint skeleton
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], (120, 120, 120), 1)

            raw = pts[8]
            smooth_pt = raw if smooth_pt is None else (
                int(smooth_pt[0] * SMOOTH + raw[0] * (1 - SMOOTH)),
                int(smooth_pt[1] * SMOOTH + raw[1] * (1 - SMOOTH)),
            )
            cursor = smooth_pt

            _thumb, idx, mid, ring, pinky = fingers_up(pts)
            four_up = idx and mid and ring and pinky
            fist = not (idx or mid or ring or pinky)

            # ---- gesture decision ----
            if fist:
                mode = "FIST - hold to CLEAR"
                prev = None
                if fist_start is None:
                    fist_start = time.time()
                held = time.time() - fist_start
                cx, cy = pts[9]
                cv2.ellipse(frame, (cx, cy), (45, 45), -90, 0,
                            int(360 * min(held / CLEAR_HOLD_SECONDS, 1)), (0, 0, 255), 6)
                if held >= CLEAR_HOLD_SECONDS:
                    canvas[:] = 0
                    fist_start = time.time() + 1.0   # cooldown
            else:
                fist_start = None

                if four_up:
                    mode = "ERASE"
                    cv2.circle(frame, cursor, ERASER, (255, 255, 255), 2)
                    if prev is not None:
                        cv2.line(canvas, prev, cursor, (0, 0, 0), ERASER * 2)
                    cv2.circle(canvas, cursor, ERASER, (0, 0, 0), cv2.FILLED)
                    prev = cursor

                elif idx and mid and not ring and not pinky:
                    mode = "SELECT"
                    prev = None
                    mx = (pts[8][0] + pts[12][0]) // 2
                    my = (pts[8][1] + pts[12][1]) // 2
                    cursor = (mx, my)
                    cv2.circle(frame, cursor, 14, (0, 255, 255), 3)
                    if my < BAR_H:
                        hover = min(mx // (w // len(PALETTE)), len(PALETTE) - 1)
                        selected = hover

                elif idx and not mid and not ring and not pinky:
                    mode = "DRAW"
                    color = PALETTE[selected][1]
                    if PALETTE[selected][0] == "Eraser":
                        if prev is not None:
                            cv2.line(canvas, prev, cursor, (0, 0, 0), brush * 4)
                        cv2.circle(canvas, cursor, brush * 2, (0, 0, 0), cv2.FILLED)
                        cv2.circle(frame, cursor, brush * 2, (255, 255, 255), 2)
                    else:
                        if cursor[1] > BAR_H:     # don't draw on the palette bar
                            if prev is not None:
                                cv2.line(canvas, prev, cursor, color, brush, cv2.LINE_AA)
                            else:
                                cv2.circle(canvas, cursor, brush // 2, color, cv2.FILLED)
                        cv2.circle(frame, cursor, brush + 4, color, 2)
                    prev = cursor
                else:
                    mode = "HOVER"
                    prev = None
        else:
            prev = None
            smooth_pt = None
            fist_start = None

        frame = merge_canvas(frame, canvas)
        draw_palette(frame, selected, hover)

        # status
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1 / max(now - last_t, 1e-3))
        last_t = now
        cv2.putText(frame, f"FPS: {int(fps)}", (w - 120, BAR_H + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Mode: {mode}   Brush: {brush}", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "[+/-] size  [s] save  [c] clear  [q] quit", (w - 480, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Air Canvas", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            canvas[:] = 0
        elif key in (ord("+"), ord("=")):
            brush = min(brush + 2, 40)
        elif key in (ord("-"), ord("_")):
            brush = max(brush - 2, 2)
        elif key == ord("s"):
            saved += 1
            # transparent-looking white-background drawing
            out = 255 - np.zeros_like(canvas)
            m = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 10
            out[m] = canvas[m]
            cv2.imwrite(f"drawing_{saved}.png", out)
            print(f"Saved drawing_{saved}.png")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()


if __name__ == "__main__":
    main()
