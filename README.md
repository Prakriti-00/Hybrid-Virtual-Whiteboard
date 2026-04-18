# Virtual Whiteboard using Hand Gestures

This project implements a real-time virtual whiteboard that allows users to draw using hand gestures captured through a webcam.

## Features

* Draw using index finger
* Stop drawing using fist gesture
* Clear canvas using "C" gesture
* Real-time performance using MediaPipe and ResNet-18

demo-
* https://drive.google.com/file/d/1IC16V5FjDZnMJdu3sFFDzVjJLL4xCSJj/view?usp=sharing

## Approach

The system uses a hybrid architecture:

* MediaPipe is used for fingertip tracking
* ResNet-18 is used for gesture classification
* A simple rule-based check is used to improve stability

## Comparison

Two versions are included:

* `dl_only_baseline.py`
  Uses only the deep learning model for gesture classification. This version is less stable in real-time.

* `finalhybrid.py`
  Uses a hybrid approach combining MediaPipe and deep learning, resulting in improved stability and performance.

## How to Run

Make sure all dependencies are installed, then run:

```bash
python finalhybrid.py
```

## Files

* `finalhybrid.py` – main implementation
* `dl_only_baseline.py` – baseline version using only deep learning
* `gesture_model_3class.pth` – trained model
* `pipeline.png` – system architecture
* `loss_curve.png` – training results

## Notes

* The system works best in good lighting conditions
* A webcam is required for input
