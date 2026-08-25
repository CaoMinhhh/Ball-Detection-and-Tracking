"""
=============================================================================
EMPIRICAL ANALYSIS & CALIBRATION TOOL
=============================================================================
Purpose:
    1. To analyze the lifespan and confidence distribution of ByteTrack IDs,
       proving why ByteTrack fragments tracking for small, fast objects.
    2. To calibrate the spatial boundary (Y-coordinate) for field filtering.

Usage:
    - Run the tracker analysis: Adjust START_SECOND and DURATION_SECOND.
    - Run the boundary calibration: Uncomment `calibrate_field_boundary()` at 
      the bottom.
=============================================================================
"""

import cv2
import argparse
from collections import defaultdict
from ultralytics import YOLO

# --- CONFIGURATION ---
BALL_MODEL_PATH = "runs/detect/train/weights/best.pt"
VIDEO_PATH = "clip_test.mp4"
TRACKER_CONFIG = "Ball_bytetrack.yaml"
CALIBRATION_IMAGE = "debug_ball_frame.jpg"

BALL_CONF = 0.05
INFER_IMGSZ = 1280

START_SECOND = 5.0
DURATION_SECOND = 8.0 


def analyze_tracker_fragmentation():
    """Analyzes how ByteTrack assigns IDs to the ball over a short sequence."""
    print("[*] Loading YOLO model for empirical analysis...")
    ball_model = YOLO(BALL_MODEL_PATH)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[!] Error: Cannot open video '{VIDEO_PATH}'")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_MSEC, START_SECOND * 1000)

    track_lifespan = defaultdict(int)      # Count of frames per Track ID
    track_conf_values = defaultdict(list)  # List of confidences per Track ID

    n_frames = int(DURATION_SECOND * fps)
    frame_count = 0

    print(f"[*] Analyzing {n_frames} frames ({DURATION_SECOND}s sequence)...")
    print(f"[*] Conf threshold: {BALL_CONF} | Tracker: {TRACKER_CONFIG}\n")

    while frame_count < n_frames:
        ret, frame = cap.read()
        if not ret:
            break

        results = ball_model.track(
            frame, persist=True, tracker=TRACKER_CONFIG,
            conf=BALL_CONF, imgsz=INFER_IMGSZ, verbose=False
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            ids = results[0].boxes.id.int().cpu().tolist()
            confs = results[0].boxes.conf.cpu().tolist()
            for tid, c in zip(ids, confs):
                track_lifespan[tid] += 1
                track_conf_values[tid].append(c)

        frame_count += 1
        if frame_count % int(fps) == 0:
            print(f"    -> Processed {frame_count}/{n_frames} frames...")

    cap.release()

    # Print Analysis Report
    print("\n" + "="*65)
    print(f"{'Track ID':<12}{'Lifespan (Frames)':<20}{'Avg Conf':<15}{'Conf Min-Max'}")
    print("-" * 65)
    
    for tid in sorted(track_lifespan.keys(), key=lambda t: -track_lifespan[t]):
        confs = track_conf_values[tid]
        avg_c = sum(confs) / len(confs)
        min_c, max_c = min(confs), max(confs)
        print(f"{tid:<12}{track_lifespan[tid]:<20}{avg_c:<15.3f}{min_c:.3f} - {max_c:.3f}")

    print("-" * 65)
    print(f"[+] Total unique Track IDs generated: {len(track_lifespan)}")
    print("[i] CONCLUSION:")
    print("    - Long lifespan -> Highly likely to be the actual ball.")
    print("    - Short lifespan (1-3 frames) -> Transient noise (lines, flags, shoes).")
    print("    - Excessive IDs prove the necessity of our Custom SOT Heuristic.")
    print("=" * 65 + "\n")


def calibrate_field_boundary():
    """Interactive tool to find the Y-coordinate for spatial filtering."""
    print(f"[*] Opening calibration image: {CALIBRATION_IMAGE}")
    img = cv2.imread(CALIBRATION_IMAGE)
    if img is None:
        print(f"[!] Error: Image '{CALIBRATION_IMAGE}' not found.")
        return

    def show_coord(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"[+] Clicked Coordinate -> X: {x}, Y: {y}")
            # Draw a temporary line to visualize the boundary
            temp_img = img.copy()
            cv2.line(temp_img, (0, y), (temp_img.shape[1], y), (0, 0, 255), 2)
            cv2.putText(temp_img, f"Y = {y}", (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Field Boundary Calibration (Click to measure)", temp_img)

    print("[i] Instruction: Click on the top edge of the grass field.")
    print("[i] Press any key on the image window to exit.")
    
    cv2.imshow("Field Boundary Calibration (Click to measure)", img)
    cv2.setMouseCallback("Field Boundary Calibration (Click to measure)", show_coord)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # 1. Run the ByteTrack fragmentation analysis
    analyze_tracker_fragmentation()
    
    # 2. Uncomment the line below to run the interactive Y-coordinate calibration tool
    # calibrate_field_boundary()