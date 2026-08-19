import os
import glob
import shutil

# --- 1. THIẾT LẬP ĐƯỜNG DẪN ---
# Chú ý: Trỏ vào thư mục sau khi bạn đã giải nén train.zip
tracking_dir = "E:/SoccerNet/soccernet_data/tracking/train" 
yolo_img_dir = "E:/SoccerNet/soccernet_yolo/images/train"
yolo_lbl_dir = "E:/SoccerNet/soccernet_yolo/labels/train"

os.makedirs(yolo_img_dir, exist_ok=True)
os.makedirs(yolo_lbl_dir, exist_ok=True)

# Hàm Toán học: Chuyển MOT (Top-Left) sang YOLO (Center)
def convert_to_yolo(img_w, img_h, x, y, w, h):
    center_x = x + (w / 2.0)
    center_y = y + (h / 2.0)
    return (max(0.0, min(1.0, center_x / img_w)), 
            max(0.0, min(1.0, center_y / img_h)), 
            w / img_w, 
            h / img_h)

# Quét tất cả các thư mục trận đấu (SNMOT060, SNMOT061...)
snmot_folders = glob.glob(os.path.join(tracking_dir, "SNMOT*"))
print(f"Phát hiện {len(snmot_folders)} trận đấu. Bắt đầu cào dữ liệu...\n")

total_frames_with_ball = 0

for folder in snmot_folders:
    match_name = os.path.basename(folder)
    print(f"⏳ Đang xử lý trận: {match_name}...")
    
    # BƯỚC 2: Đọc kích thước ảnh từ seqinfo.ini
    img_w, img_h = 1920, 1080 # Mặc định
    try:
        with open(os.path.join(folder, "seqinfo.ini"), "r") as f:
            for line in f:
                if "imWidth" in line: img_w = int(line.split('=')[1])
                if "imHeight" in line: img_h = int(line.split('=')[1])
    except FileNotFoundError:
        pass
        
    # BƯỚC 3: Đọc gameinfo.ini xem Track_ID nào là quả bóng
    ball_track_ids = []
    try:
        with open(os.path.join(folder, "gameinfo.ini"), "r") as f:
            for line in f:
                # VD: trackletID_2= ball;
                if "trackletID_" in line and "ball" in line.lower():
                    # Lấy con số ID
                    t_id = line.split("=")[0].split("_")[1].strip()
                    ball_track_ids.append(t_id)
    except FileNotFoundError:
        pass

    if not ball_track_ids:
        print(f"   -> Không tìm thấy bóng trong {match_name}, bỏ qua.")
        continue

    # BƯỚC 4: Duyệt Ground Truth (gt.txt)
    gt_path = os.path.join(folder,"gt", "gt.txt")
    if not os.path.exists(gt_path):
        continue
    
    # Dictionary Gom nhóm tọa độ theo từng Frame
    # frame_dict = { "1": ["0 0.5 0.5 0.1 0.1", ...], "2": [...] }
    frame_dict = {}
    
    with open(gt_path, "r") as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6: continue
            
            frame_id = int(parts[0])
            track_id = parts[1].strip()
            
            # Nếu ID này thuộc về quả bóng
            if track_id in ball_track_ids:
                x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                nx, ny, nw, nh = convert_to_yolo(img_w, img_h, x, y, w, h)
                
                yolo_str = f"0 {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"
                
                if frame_id not in frame_dict:
                    frame_dict[frame_id] = []
                frame_dict[frame_id].append(yolo_str)

    # BƯỚC 5: Copy Ảnh và Ghi file TXT
    img_folder = os.path.join(folder, "img1") 
        
    for frame_id, yolo_lines in frame_dict.items():
        orig_img_name = f"{frame_id:06d}.jpg"
        orig_img_path = os.path.join(img_folder, orig_img_name)
        
        if os.path.exists(orig_img_path):
            new_prefix = f"{match_name}_{frame_id:06d}"
            
            # Ghi TXT
            with open(os.path.join(yolo_lbl_dir, f"{new_prefix}.txt"), "w") as f:
                f.write("\n".join(yolo_lines))
                
            # Copy ảnh
            shutil.copy2(orig_img_path, os.path.join(yolo_img_dir, f"{new_prefix}.jpg"))
            total_frames_with_ball += 1

print(f"\n🎉 HOÀN TẤT! Đã đóng gói thành công {total_frames_with_ball} cặp Ảnh + Tọa độ cho YOLO.")