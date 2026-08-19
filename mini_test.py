# """
# Debug tracking qua nhiều frame liên tiếp (không phải 1 frame tĩnh) để xác định lại ngưỡng.
# Lý do: .track() với persist=True cần chuỗi frame để ByteTrack quyết định có giữ track hay không,
# test 1 frame đơn không phản ánh đúng hành vi thật của new_track_thresh / track_buffer.

# Cách dùng:
#     Sửa START_SECOND, DURATION_SECOND cho đúng đoạn bạn muốn kiểm tra
#     (nên chọn đoạn có cả bóng thật lẫn khu vực dễ nhiễu như cờ trọng tài, vạch biên...).
#     Chạy: python debug_ball_track_sequence.py
# """

# import cv2
# from collections import defaultdict
# from ultralytics import YOLO

# BALL_MODEL_PATH = r"E:/SoccerNet/runs/detect/train-4/weights/best.pt"
# VIDEO_PATH = "clip_test.mp4"
# TRACKER_CONFIG = "Ball_bytetrack.yaml"
# BALL_CONF = 0.05
# INFER_IMGSZ = 1280

# START_SECOND = 5.0
# DURATION_SECOND = 8.0     # chạy qua ~8 giây để đủ thấy track nào sống lâu, track nào chớp nhoáng

# ball_model = YOLO(BALL_MODEL_PATH)

# cap = cv2.VideoCapture(VIDEO_PATH)
# fps = cap.get(cv2.CAP_PROP_FPS) or 25
# cap.set(cv2.CAP_PROP_POS_MSEC, START_SECOND * 1000)

# track_lifespan = defaultdict(int)      # đếm số frame mỗi track ID xuất hiện
# track_conf_values = defaultdict(list)  # lưu lại conf của từng track ID qua các frame

# n_frames = int(DURATION_SECOND * fps)
# frame_count = 0

# print(f"--- Chạy qua {n_frames} frame ({DURATION_SECOND}s), conf={BALL_CONF}, tracker={TRACKER_CONFIG} ---\n")

# while frame_count < n_frames:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = ball_model.track(
#         frame, persist=True, tracker=TRACKER_CONFIG,
#         conf=BALL_CONF, imgsz=INFER_IMGSZ, verbose=False
#     )

#     if results[0].boxes.id is not None:
#         ids = results[0].boxes.id.int().cpu().tolist()
#         confs = results[0].boxes.conf.cpu().tolist()
#         for tid, c in zip(ids, confs):
#             track_lifespan[tid] += 1
#             track_conf_values[tid].append(c)

#     frame_count += 1

# cap.release()

# print(f"{'Track ID':<10}{'Số frame sống':<16}{'Conf trung bình':<18}{'Conf min-max'}")
# print("-" * 60)
# for tid in sorted(track_lifespan.keys(), key=lambda t: -track_lifespan[t]):
#     confs = track_conf_values[tid]
#     avg_c = sum(confs) / len(confs)
#     print(f"{tid:<10}{track_lifespan[tid]:<16}{avg_c:<18.3f}{min(confs):.3f}-{max(confs):.3f}")

# print(f"\nTổng số track ID khác nhau xuất hiện: {len(track_lifespan)}")
# print("Track sống LÂU (nhiều frame) + xuyên suốt => rất có thể là bóng thật.")
# print("Track sống NGẮN (1-3 frame) => rất có thể là nhiễu (cờ, vạch, số áo...).")
# print("Dựa vào đây để quyết định: có cần tăng new_track_thresh trong ball_bytetrack.yaml không,")
# print("hoặc thêm điều kiện lọc trong script chính: bỏ qua track có lifespan quá ngắn.")

import cv2
img = cv2.imread("debug_ball_frame.jpg")  # lấy 1 frame từ video của bạn
def show_coord(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"y = {y}")
cv2.imshow("Click vao mep duoi bien quang cao", img)
cv2.setMouseCallback("Click vao mep duoi bien quang cao", show_coord)
cv2.waitKey(0)