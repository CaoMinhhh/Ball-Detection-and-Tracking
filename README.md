# Custom Heuristic Soccer Ball & Player Tracking

This repository contains the source code for a robust end-to-end pipeline that detects and tracks a soccer ball and players in broadcast football videos. 

## Key Contributions & Methodology
Tracking small, fast-moving objects like a soccer ball is challenging for standard Multi-Object Tracking (MOT) algorithms. Traditional trackers (like ByteTrack or SORT) rely heavily on frame-to-frame Intersection over Union (IoU). For a ball moving at high speeds, the IoU between consecutive frames often drops to zero, causing severe ID fragmentation (e.g., a single ball assigned to 43 different IDs).

To solve this, our pipeline implements a **Hybrid Tracking Architecture**:
1. **For the Ball (Custom SOT Heuristic):** We discard standard MOT trackers and implement a custom distance-and-confidence-based penalty function. It assumes only one ball is on the field, utilizing spatial priors and anti-drift recovery mechanisms to maintain a single continuous trajectory.
2. **For Players (ByteTrack):** We utilize the standard ByteTrack algorithm, as players are large enough to maintain stable bounding box overlaps.

## Project Structure

* `test.py`: **[Main]** The core inference script. Runs YOLOv8 detection, applies the Custom Heuristic Tracker for the ball, ByteTrack for players, and renders the output video with a motion fade trail.
* `train.py`: Training pipeline configuration for fine-tuning the YOLOv8 Nano model.
* `mini_test.py`: An empirical analysis and calibration tool used to measure track lifespans and extract spatial field boundaries.
* `cutting_video.py`: A lightweight utility to trim large match videos into manageable ~90-second test clips.
* `convert_to_yolo.py`: Data preparation script to convert MOT dataset formats into YOLO normalized coordinates.
* `soccer.yaml`: Dataset configuration file for YOLOv8.
* `Ball_bytetrack.yaml`: Custom ByteTrack configuration heavily modified for small, fast-moving objects.

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/CaoMinhhh/Ball-Detection-and-Tracking.git
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
## How to use
1. Run the Main Tracker:
Ensure you have a test video (e.g., clip_test.mp4) and your trained weights (best.pt) in the correct directories. Open test.py, verify the configuration paths at the top of the file, and run:
   ```bash
   python test.py
_Press 'q' at any time to interrupt the processing_

2. Run the Empirical Analysis Tool:
If you want to debug track fragmentation or calibrate the field boundary (Y-coordinate):
   ```bash
   python mini_test.py

## Configuration Adjustments:
Inside `test.py`, you can toggle features or adjust tracking heuristics:
- Set `ENABLE_PLAYER_TRACKING = True` to render player bounding boxes and IDs.
- Modify `BALL_MAX_JUMP_DIST` or `BALL_DIST_PENALTY_WEIGHT` to adapt the tracker to different camera zoom levels.

## Link video demo: https://youtu.be/tCL2UsQliEA
