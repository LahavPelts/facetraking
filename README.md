# Face Tracking

A computer-only face tracking project using [MediaPipe](https://github.com/google/mediapipe) and [cvzone](https://github.com/cvzone/cvzone) for real-time face detection through a webcam. The app detects your face, draws a bounding box around it, and shows how far it is from the center of the frame (`dx`, `dy`) with a reference crosshair — no external hardware required.

## Setup

Clone the repository:

```
git clone https://github.com/LahavPelts/facetraking.git
cd facetraking
```

Create a virtual environment:

```
python -m venv env
```

Activate it:

- **Windows:** `env\Scripts\activate`
- **macOS/Linux:** `source env/bin/activate`

Install dependencies:

```
pip install -r requirements.txt
```

## Running

Combined face + hand tracking:
```
python main.py
```

Face mesh only:
```
python Facemeshmodule.py
```

Hand tracking only:
```
python HandtrackingModule.py
```

Face + hand tracking with Arduino serial output:
```
python trackingwiththreads.py
```

Press **q** to quit any of the above.

## Other files

- `main.py` — combines face and hand tracking in one script
- `Facemeshmodule.py` — reusable 468-point face mesh detector class
- `HandtrackingModule.py` — reusable hand landmark detector class
- `mesh.py` — minimal face mesh demo using cvzone

## Notes

This project previously sent face position data over serial to an Arduino to drive pan/tilt servos. That hardware dependency has been removed — `dx`/`dy` are now just displayed on screen, but could be repurposed to drive something else (a digital pan/crop, a cursor, etc.) if you want to extend it.