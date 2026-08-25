import os
import glob
import shutil

# --- 1. DIRECTORY CONFIGURATION ---
# Note: Update these paths relative to your current working directory
tracking_dir = "./soccernet_data/tracking/train" 
yolo_img_dir = "./soccernet_yolo/images/train"
yolo_lbl_dir = "./soccernet_yolo/labels/train"

os.makedirs(yolo_img_dir, exist_ok=True)
os.makedirs(yolo_lbl_dir, exist_ok=True)


def convert_to_yolo(img_w, img_h, x, y, w, h):
    """
    Convert bounding box from MOT format (Top-Left x, Top-Left y, width, height)
    to YOLO format (Center x, Center y, width, height) normalized to [0, 1].
    """
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    return (max(0.0, min(1.0, center_x / img_w)), 
            max(0.0, min(1.0, center_y / img_h)), 
            w / img_w, 
            h / img_h)

# Scan all match folders (e.g., SNMOT060, SNMOT061...)
snmot_folders = glob.glob(os.path.join(tracking_dir, "SNMOT*"))
print(f"[*] Found {len(snmot_folders)} match folders. Starting conversion...\n")

total_frames_with_ball = 0

for folder in snmot_folders:
    match_name = os.path.basename(folder)
    print(f"[>] Processing match: {match_name}...")
    
    # --- STEP 2: Read image dimensions from seqinfo.ini ---
    img_w, img_h = 1920, 1080 # Default fallback dimensions
    try:
        with open(os.path.join(folder, "seqinfo.ini"), "r") as f:
            for line in f:
                if "imWidth" in line: img_w = int(line.split('=')[1])
                if "imHeight" in line: img_h = int(line.split('=')[1])
    except FileNotFoundError:
        pass
        
# --- STEP 3: Identify ball Track_ID from gameinfo.ini ---
    ball_track_ids = []
    try:
        with open(os.path.join(folder, "gameinfo.ini"), "r") as f:
            for line in f:
                # Example format: trackletID_2= ball;
                if "trackletID_" in line and "ball" in line.lower():
                    t_id = line.split("=")[0].split("_")[1].strip()
                    ball_track_ids.append(t_id)
    except FileNotFoundError:
        pass

    if not ball_track_ids:
        print(f"    -> No ball track ID found in {match_name}. Skipping.")
        continue

# --- STEP 4: Parse Ground Truth (gt.txt) ---
    gt_path = os.path.join(folder,"gt", "gt.txt")
    if not os.path.exists(gt_path):
        continue
    
    # Dictionary to group coordinates by frame ID
    # Format: { 1: ["0 0.5 0.5 0.1 0.1", ...], 2: [...] }
    frame_dict = {}
    
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            
            frame_id = int(parts[0])
            track_id = parts[1].strip()
            
            # Filter and process only ball bounding boxes
            if track_id in ball_track_ids:
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                nx, ny, nw, nh = convert_to_yolo(img_w, img_h, x, y, w, h)
                
                yolo_str = f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"
                
                if frame_id not in frame_dict:
                    frame_dict[frame_id] = []
                frame_dict[frame_id].append(yolo_str)

# --- STEP 5: Copy Images and Write YOLO txt files ---
    img_folder = os.path.join(folder, "img1") 
        
    for frame_id, yolo_lines in frame_dict.items():
        orig_img_name = f"{frame_id:06d}.jpg"
        orig_img_path = os.path.join(img_folder, orig_img_name)
        
        if os.path.exists(orig_img_path):
            new_prefix = f"{match_name}_{frame_id:06d}"
            
            # Write YOLO format labels
            with open(os.path.join(yolo_lbl_dir, f"{new_prefix}.txt"), "w") as f:
                f.write("\n".join(yolo_lines))
                
            # Copy corresponding image
            shutil.copy2(orig_img_path, os.path.join(yolo_img_dir, f"{new_prefix}.jpg"))
            total_frames_with_ball += 1

print(f"\n[+] DONE! Successfully generated {total_frames_with_ball} image-label pairs for YOLO training.")