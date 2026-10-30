# Hand-Tracking AR User Interface

Real-time hand tracking with MediaPipe + OpenCV.

- **BASIC mode**: green skeleton and a green circle on the index fingertip (landmark 8)
- **AR mode**: cyan rotating HUD around each palm, finger count, thumb-index pinch distance

## Setup (Python 3.9+)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
python main.py
```

## Controls

| Key | Action |
|-----|--------|
| m   | Switch BASIC / AR mode |
| s   | Save screenshot |
| q   | Quit |

## Troubleshooting

- Black window or no camera: change `CAM_INDEX = 0` to `1` in `main.py`.
- Low FPS: reduce `FRAME_W`/`FRAME_H` or set `MAX_HANDS = 1`.
- On first run the script downloads `hand_landmarker.task` (needs internet once).

## Ideas to extend

- Virtual mouse (`pyautogui`) using landmark 8
- Volume control with pinch distance (`pycaw`)
- Gesture-based air drawing canvas

## Air Canvas

```bash
python air_canvas.py
```

| Gesture | Action |
|---------|--------|
| Index finger only | Draw |
| Index + middle fingers | Select color (hover over the top bar) |
| Open palm | Erase |
| Fist (hold 1 sec) | Clear canvas |

Keys: `+` / `-` brush size, `s` save, `c` clear, `q` quit.
