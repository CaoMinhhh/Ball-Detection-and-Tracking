"""
=============================================================================
CUSTOM BALL & PLAYER TRACKING PIPELINE (FINAL SUBMISSION)
=============================================================================
Prerequisites:
    pip install ultralytics opencv-python

Usage:
    1. Ensure YOLOv8 weights (BALL_MODEL_PATH and PERSON_MODEL_PATH) are correct.
    2. Set INPUT_VIDEO to your target clip.
    3. Run: `python test.py` (Press 'q' to quit early).

METHODOLOGY & DESIGN DECISIONS (Heuristic Tracking Algorithm):
    1. Single-Object Tracking (SOT) Regression: Replaced ByteTrack for the ball. 
       ByteTrack relies on frame-to-frame IoU, which fails for small, fast-moving 
       objects (IoU drops to 0, causing severe ID fragmentation - e.g., 43 IDs 
       in 200 frames).
    2. Multi-Object Tracking (MOT) Retention: Kept ByteTrack for players, as 
       they are large enough to maintain stable IoU and require ID differentiation.
    3. Shape & Spatial Prior Filtering: Candidates are filtered by bounding box 
       area, aspect ratio, and field coordinates to eliminate geometric false 
       positives (e.g., shoes, corner flags, audience).
    4. Distance-Confidence Hybrid Scoring: The candidate selection metric penalizes 
       distance while rewarding YOLO confidence. This prevents the tracker from 
       latching onto weak nearby noise while ignoring high-confidence actual balls 
       slightly further away.
    5. Anti-Drift Mechanism (Stuck Recovery): If track confidence remains below 
       a threshold for consecutive frames, the tracker actively drops the trajectory 
       to prevent locking onto static background noise.

KNOWN LIMITATIONS:
    - Trajectory gaps occur during severe motion blur or complete occlusion.
    - Spatial boundary (BALL_FIELD_TOP_Y) assumes a relatively static broadcast 
      camera angle; heavy pan/zoom may require dynamic thresholding.
=============================================================================
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------- 1. CONFIGURATION -----------------------
# Paths (Use relative paths for portability)
BALL_MODEL_PATH = "runs/detect/train/weights/best.pt"
PERSON_MODEL_PATH = "yolov8n.pt"           # Pretrained COCO model
INPUT_VIDEO = "clip_test.mp4"
OUTPUT_VIDEO = "tracked_output.mp4"

# Feature Toggles
ENABLE_PLAYER_TRACKING = False             # Set to True to enable ByteTrack for players

# Ball Detection & Tracking Thresholds
BALL_CONF = 0.05                           # Extremely low conf to catch motion blur
BALL_INFER_IMGSZ = 1280                    # High resolution inference to preserve small pixels
BALL_MAX_JUMP_DIST = 100                   # px: Max allowed Euclidean distance between frames
BALL_FALLBACK_MIN_CONF = 0.3               # Min conf to accept a long-distance jump
BALL_MIN_AREA = 30                         # px^2: Min bounding box area
BALL_MAX_AREA = 1600                       # px^2: Max bounding box area (filters out shoes)
BALL_MIN_ASPECT = 0.5                      # Min w/h ratio (ball is nearly square)
BALL_MAX_ASPECT = 2.0                      # Max w/h ratio
BALL_FIELD_TOP_Y = 330                     # px: Upper boundary to filter audience detections
BALL_DIST_PENALTY_WEIGHT = 0.6             # Weight for distance penalty in selection scoring
BALL_STUCK_CONF_THRESH = 0.15              # Threshold to detect static noise lock
BALL_STUCK_FRAME_LIMIT = 8                 # Consecutive low-conf frames before dropping track

# Player Tracking Configurations
PERSON_CONF = 0.35
PERSON_DETECT_EVERY_N_FRAMES = 2           # Stride to reduce compute load

# Visualization
TRAIL_LEN = 30                             # Number of history points for the fade trail
# ----------------------------------------------------------------

def main():
    print("[*] Loading YOLO models...")
    ball_model = YOLO(BALL_MODEL_PATH)
    if ENABLE_PLAYER_TRACKING:
        person_model = YOLO(PERSON_MODEL_PATH)

    ball_trail = []             # Stores (x, y) tuples or None (break markers)
    ball_last_pos = None
    ball_low_conf_streak = 0
    last_person_boxes = []

    def trim_trail():
        """Maintains the trail buffer size."""
        while len(ball_trail) > TRAIL_LEN:
            ball_trail.pop(0)

    print(f"[*] Opening video: {INPUT_VIDEO}")
    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise FileNotFoundError(f"[!] Error: Cannot open video '{INPUT_VIDEO}'")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

    frame_idx = 0
    print("[>] Processing frames (Press 'q' to stop)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated = frame.copy()

        # ================= A. BALL PROCESSING =================
        ball_results = ball_model.predict(frame, conf=BALL_CONF, imgsz=BALL_INFER_IMGSZ, verbose=False)
        boxes = ball_results[0].boxes

        if len(boxes) > 0:
            xywh = boxes.xywh.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            # Geometric & Spatial Filtering
            bw, bh = xywh[:, 2], xywh[:, 3]
            area = bw * bh
            aspect = bw / np.maximum(bh, 1e-6)
            shape_mask = (
                (area >= BALL_MIN_AREA) & (area <= BALL_MAX_AREA) &
                (aspect >= BALL_MIN_ASPECT) & (aspect <= BALL_MAX_ASPECT) &
                (xywh[:, 1] >= BALL_FIELD_TOP_Y)
            )
            xywh, xyxy, confs = xywh[shape_mask], xyxy[shape_mask], confs[shape_mask]
        else:
            xywh, xyxy, confs = np.empty((0, 4)), np.empty((0, 4)), np.empty((0,))

        if len(xywh) > 0:
            # Candidate Selection: Score = Confidence - Distance_Penalty
            if ball_last_pos is not None:
                dists = np.linalg.norm(xywh[:, :2] - np.array(ball_last_pos), axis=1)
                scores = confs - BALL_DIST_PENALTY_WEIGHT * (dists / BALL_MAX_JUMP_DIST)
            else:
                scores = confs
                
            display_idx = int(np.argmax(scores))
            cx, cy = xywh[display_idx][0], xywh[display_idx][1]
            display_conf = confs[display_idx]

            # Continuity Decision
            should_connect = False
            if ball_last_pos is not None:
                dist_to_last = np.linalg.norm(np.array([cx, cy]) - np.array(ball_last_pos))
                should_connect = (dist_to_last <= BALL_MAX_JUMP_DIST) or (display_conf >= BALL_FALLBACK_MIN_CONF)

            # Anti-Drift (Stuck Recovery) Mechanism
            ball_low_conf_streak = ball_low_conf_streak + 1 if display_conf < BALL_STUCK_CONF_THRESH else 0

            if ball_low_conf_streak >= BALL_STUCK_FRAME_LIMIT:
                ball_last_pos = None
                ball_low_conf_streak = 0
                ball_trail.append(None)
                trim_trail()
                # Skip box drawing for this frame to break the drift lock
            else:
                if not should_connect and ball_last_pos is not None:
                    ball_trail.append(None)
                    trim_trail()

                ball_last_pos = (cx, cy)
                ball_trail.append((float(cx), float(cy)))
                trim_trail()

                # Draw Bounding Box
                x1, y1, x2, y2 = xyxy[display_idx].astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(annotated, f"Ball {display_conf:.2f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Draw Motion Fade Trail
            for i in range(1, len(ball_trail)):
                p1, p2 = ball_trail[i - 1], ball_trail[i]
                if p1 is None or p2 is None:
                    continue
                alpha = i / len(ball_trail)
                thickness = max(1, int(4 * alpha))
                color = (0, int(255 * alpha), int(255 * alpha))
                cv2.line(annotated, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
        
        # ================= B. PLAYER PROCESSING (Optional) =================
        if ENABLE_PLAYER_TRACKING:
            if frame_idx % PERSON_DETECT_EVERY_N_FRAMES == 0:
                person_results = person_model.track(
                    frame, persist=True, tracker='bytetrack.yaml', conf=PERSON_CONF, classes=[0], verbose=False
                )
                last_person_boxes = []
                if person_results[0].boxes is not None and person_results[0].boxes.id is not None:
                    p_boxes = person_results[0].boxes.xyxy.cpu().numpy()
                    p_ids = person_results[0].boxes.id.int().cpu().tolist()
                    last_person_boxes = list(zip(p_boxes, p_ids))

            for box, pid in last_person_boxes:
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated, f"ID:{pid}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # ================= C. OUTPUT & PREVIEW =================
        writer.write(annotated)
        cv2.imshow('Custom Heuristic Tracking Pipeline', annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[!] Processing interrupted by user.")
            break

        frame_idx += 1

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\n[+] DONE! Tracked output saved successfully at: {OUTPUT_VIDEO}")

if __name__ == '__main__':
    main()