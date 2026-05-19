# AuraBeat

A real-time, gesture-controlled music application that turns hand movements captured through a standard webcam into musical notes. Built with Python, MediaPipe, Kivy, OpenCV, and pyo.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![Kivy](https://img.shields.io/badge/Kivy-2.3.1-brightgreen) ![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.33-orange) ![pyo](https://img.shields.io/badge/pyo-1.0.5-purple)

---

## Overview

AuraBeat uses computer vision to track your hands in real time and maps each finger to a musical note. Press a finger down past the knuckle line and the note plays — hold it there and it sustains, lift it and it releases. Both hands are tracked simultaneously, giving you 10 independent notes at once.

Recorded sessions are displayed in a Synthesia-style falling-note piano roll and can be played back with live input automatically muted.

---

## Features

- **Knuckle-line press detection** — notes trigger when a fingertip crosses the MCP knuckle line into the palm, and sustain until released
- **Two-hand play** — up to 10 simultaneous notes across both hands
- **Sliding scale window** — fist gesture shifts your note window up or down the scale; thumbs-up shifts it the other way
- **Custom pitch assignment** — click any finger label to assign it any note on an 88-key keyboard
- **Session recording** — press Record, play, press again to stop; notes are committed to the piano roll
- **Falling-note playback** — notes fall toward the keyboard strike line in sync with audio
- **Mute toggle** — suppress live input independently of playback
- **Transport controls** — Play, Pause, Loop wired to the playback engine
- **Expanded piano roll** — press `F` for a full 88-key popup view

---

## Requirements

- Windows, macOS, or Linux
- Standard webcam (720p or higher recommended)
- Audio output device (speakers or headphones)
- **Python 3.10** — required for pyo on Windows (pyo has no pre-built wheel for Python 3.13)

---

## Installation

### 1. Create a conda environment with Python 3.10

```bash
conda create -n aurabeat python=3.10
conda activate aurabeat
```

### 2. Install dependencies

```bash
pip install pyo kivy opencv-python mediapipe numpy
```

### 3. Download the MediaPipe hand landmark model

Download `hand_landmarker.task` from the [MediaPipe Models page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and place it at:

```
src/models/hand_landmarker.task
```

### 4. Run the app

```bash
cd src
python main.py
```

---

## Project Structure

```
AuraBeat/
├── src/
│   ├── main.py                        # App entry point
│   ├── models/
│   │   └── hand_landmarker.task       # MediaPipe model file (download separately)
│   ├── audio/
│   │   ├── engine.py                  # AudioEngine — pyo server, 10 voices
│   │   ├── bus.py                     # MasterBus gain control
│   │   ├── audio_interface.py         # Thin wrapper for playback engine
│   │   └── voices/
│   │       └── keyboard.py            # KeyboardVoice — SuperSaw + ADSR
│   ├── gestures/
│   │   ├── classifiers.py             # is_fist, is_thumbs_up, is_open, is_point
│   │   ├── finger_press.py            # Knuckle-line crossing detector
│   │   └── temporal.py                # HysteresisFlag smoothing
│   ├── hand_tracking/
│   │   ├── hands.py                   # HandTracker (MediaPipe Tasks wrapper)
│   │   └── camera.py                  # VideoController — main camera loop
│   ├── mapping/
│   │   ├── finger_ids.py              # Landmark index constants
│   │   ├── pitch_mapper.py            # (hand, finger) → MIDI pitch
│   │   └── scale_window.py            # Sliding 5-note scale window
│   ├── playback/
│   │   ├── playback_engine.py         # Clock-driven note playback
│   │   └── time_grid.py               # Time ↔ pixel coordinate conversion
│   ├── recording/
│   │   ├── note_event.py              # NoteEvent data class
│   │   ├── recorder.py                # Session recorder
│   │   └── recorder_integration.py    # Bridge: recorder → piano roll
│   └── ui/
│       ├── kv.py                      # Kivy layout string
│       └── widgets/
│           ├── __init__.py
│           ├── air_overlay.py         # 10-dot finger overlay panel
│           ├── controls.py            # PillButton, CircleIconButton
│           ├── docks.py               # UpperDock, LowerDock
│           ├── expanded_piano_roll.py # Full 88-key popup
│           ├── finger_label.py        # Clickable finger label
│           ├── key_select_dialog.py   # Custom pitch assignment dialog
│           ├── layout.py              # GradientBackground, LeftOptionsPanel
│           ├── piano_roll.py          # PianoRollPanel, NoteCanvas
│           ├── transport_controls.py  # Play/Pause/Loop buttons
│           └── video.py               # VideoFeed, GestureHUD, RootView
```

---

## Controls

| Input | Action |
|---|---|
| Finger past knuckle line | Play note (holds until released) |
| Fist (right hand) | Shift right-hand scale window up |
| Thumbs up (right hand) | Shift right-hand scale window down |
| Fist (left hand) | Shift left-hand scale window down |
| Thumbs up (left hand) | Shift left-hand scale window up |
| Click finger label | Open custom pitch assignment dialog |
| Record button | Start / stop recording |
| Play button | Play back recorded session (mutes live input) |
| Pause button | Pause playback |
| Mute button | Toggle live input on/off |
| `F` key | Open / close expanded piano roll popup |
| `F11` | Toggle fullscreen |
| `P` key | Panic — silence all active notes |
| `ESC` | Exit fullscreen |

---

## Troubleshooting

**No audio output**
- Make sure you are running in the `aurabeat` conda environment (Python 3.10)
- Run `conda activate aurabeat` before launching
- Verify pyo is installed: `python -c "import pyo; print('pyo ok')"`

**Webcam not detected**
- Check that no other application is using the camera
- Try changing `cam_index=0` to `cam_index=1` in `main.py` if you have multiple cameras

**Hand model not found**
- Make sure `hand_landmarker.task` is placed at `src/models/hand_landmarker.task`
- Download from the [MediaPipe Models page](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

**Notes not triggering**
- Ensure your hand is well-lit and fully visible in the camera frame
- Try adjusting `PRESS_THRESHOLD` and `RELEASE_THRESHOLD` in `src/gestures/finger_press.py` if detection feels too sensitive or not sensitive enough

---

## Authors

**Aarush Kapoor & Taylor Joasil**

---

## License

This project was developed as an academic project. All rights reserved.
