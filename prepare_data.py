import os
from SoccerNet.Downloader import SoccerNetDownloader

mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="soccernet_data")

mySoccerNetDownloader.downloadDataTask(task='tracking', split=['train'])

os.makedirs("soccernet_yolo/images/train", exist_ok=True)
os.makedirs("soccernet_yolo/labels/train", exist_ok=True)

def convert_to_yolo_format(image_width, image_height, bbox_x_min, bbox_y_min, bbox_w, bbox_h):
    """
    Hàm này dịch tọa độ gốc của SoccerNet sang chuẩn YOLO (0 đến 1)
    """
    # Tính tọa độ Tâm X và Tâm Y
    center_x = bbox_x_min + (bbox_w / 2.0)
    center_y = bbox_y_min + (bbox_h / 2.0)
    
    # Chuẩn hóa (Normalize) về khoảng 0 -> 1 bằng cách chia cho kích thước ảnh gốc
    norm_center_x = center_x / image_width
    norm_center_y = center_y / image_height
    norm_w = bbox_w / image_width
    norm_h = bbox_h / image_height
    
    # Đảm bảo dữ liệu không bị văng ra khỏi ảnh (clip từ 0 đến 1)
    norm_center_x = max(0.0, min(1.0, norm_center_x))
    norm_center_y = max(0.0, min(1.0, norm_center_y))
    
    # Trả về chuỗi chuẩn YOLO: <Class_ID> <Tâm X> <Tâm Y> <Rộng> <Cao>
    # Class ID = 0 (đại diện cho quả bóng)
    return f"0 {norm_center_x:.6f} {norm_center_y:.6f} {norm_w:.6f} {norm_h:.6f}"