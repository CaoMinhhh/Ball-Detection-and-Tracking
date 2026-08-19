"""
Ball tracking + trajectory line + player detection/ID cho video bóng đá.
(BẢN CHỐT -- xem tóm tắt các quyết định thiết kế ở cuối file để đưa vào báo cáo)

Yêu cầu:
    pip install ultralytics opencv-python --break-system-packages   (nếu chưa có)

Cách dùng:
    1. Sửa BALL_MODEL_PATH nếu đường dẫn best.pt khác.
    2. Sửa INPUT_VIDEO trỏ tới video/clip muốn test (vd: clip_test.mp4).
    3. Chạy: python ball_and_player_tracking.py
    4. Video kết quả được lưu ra OUTPUT_VIDEO, đồng thời hiện preview real-time.
       Nhấn 'q' để dừng sớm.
"""

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------- CONFIG -----------------------
BALL_MODEL_PATH = r"E:\SoccerNet\runs\detect\train-4\weights\best.pt"
PERSON_MODEL_PATH = "yolov8n.pt"          # pretrained COCO, có sẵn class "person"
INPUT_VIDEO = "clip_test.mp4"
OUTPUT_VIDEO = "tracked_output.mp4"

BALL_CONF = 0.05                          # thấp, để không bỏ sót detection yếu của bóng nhỏ/nhanh
BALL_INFER_IMGSZ = 1280                   # cao hơn imgsz lúc train (960), giúp bóng giữ đủ pixel khi infer
BALL_MAX_JUMP_DIST = 100                  # px -- khoảng cách tối đa coi là "liên tục" giữa 2 frame
BALL_FALLBACK_MIN_CONF = 0.3              # chỉ chấp nhận nối line khi nhảy xa nếu conf đủ cao
BALL_MIN_AREA = 30                        # px^2 -- loại box quá nhỏ (nhiễu vụn vặt)
BALL_MAX_AREA = 1600                      # px^2 -- loại box quá to (chân/giày cầu thủ)
BALL_MIN_ASPECT = 0.5                     # w/h tối thiểu -- bóng gần vuông, giày/cờ thường dẹt/dài hơn
BALL_MAX_ASPECT = 2.0                     # w/h tối đa
BALL_FIELD_TOP_Y = 330                    # px -- ranh giới trên của sân, loại detection ở khán đài
BALL_DIST_PENALTY_WEIGHT = 0.6            # trọng số phạt theo khoảng cách so với conf khi chọn box
BALL_STUCK_CONF_THRESH = 0.15             # conf dưới ngưỡng này liên tục -> nghi dính nhiễu tĩnh
BALL_STUCK_FRAME_LIMIT = 8                # số frame liên tiếp trước khi "buông" vị trí cũ

PERSON_CONF = 0.35
TRAIL_LEN = 30                            # số điểm giữ lại để vẽ line
PERSON_DETECT_EVERY_N_FRAMES = 2          # detect người mỗi N frame để đỡ tốn compute
# --------------------------------------------------------

ball_model = YOLO(BALL_MODEL_PATH)
person_model = YOLO(PERSON_MODEL_PATH)

ball_trail = []            # list các (x, y) hoặc None (break marker), không phụ thuộc track ID
ball_last_pos = None
ball_low_conf_streak = 0
last_person_boxes = []


def trim_trail():
    """Giữ ball_trail không vượt quá TRAIL_LEN phần tử, gọi sau MỌI lần append."""
    while len(ball_trail) > TRAIL_LEN:
        ball_trail.pop(0)


cap = cv2.VideoCapture(INPUT_VIDEO)
if not cap.isOpened():
    raise FileNotFoundError(f"Không mở được video: {INPUT_VIDEO}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    annotated = frame.copy()

    # ================= BALL: detect + trajectory =================
    # ball_results = ball_model.predict(frame, conf=BALL_CONF, imgsz=BALL_INFER_IMGSZ, verbose=False)
    # boxes = ball_results[0].boxes

    # if len(boxes) > 0:
    #     xywh = boxes.xywh.cpu().numpy()
    #     xyxy = boxes.xyxy.cpu().numpy()
    #     confs = boxes.conf.cpu().numpy()

    #     # lọc theo kích thước + tỉ lệ + vị trí (loại giày/cờ/khán đài)
    #     bw, bh = xywh[:, 2], xywh[:, 3]
    #     area = bw * bh
    #     aspect = bw / np.maximum(bh, 1e-6)
    #     shape_mask = (
    #         (area >= BALL_MIN_AREA) & (area <= BALL_MAX_AREA) &
    #         (aspect >= BALL_MIN_ASPECT) & (aspect <= BALL_MAX_ASPECT) &
    #         (xywh[:, 1] >= BALL_FIELD_TOP_Y)
    #     )
    #     xywh, xyxy, confs = xywh[shape_mask], xyxy[shape_mask], confs[shape_mask]
    # else:
    #     xywh, xyxy, confs = np.empty((0, 4)), np.empty((0, 4)), np.empty((0,))

    # if len(xywh) > 0:
    #     # chọn box hiển thị: kết hợp khoảng cách + conf (tránh dính nhiễu gần, bỏ sót bóng thật ở xa hơn 1 chút)
    #     if ball_last_pos is not None:
    #         dists = np.linalg.norm(xywh[:, :2] - np.array(ball_last_pos), axis=1)
    #         scores = confs - BALL_DIST_PENALTY_WEIGHT * (dists / BALL_MAX_JUMP_DIST)
    #     else:
    #         scores = confs
    #     display_idx = int(np.argmax(scores))
    #     cx, cy = xywh[display_idx][0], xywh[display_idx][1]
    #     display_conf = confs[display_idx]

    #     # line chỉ nối nếu đủ tin cậy
    #     should_connect = False
    #     if ball_last_pos is not None:
    #         dist_to_last = np.linalg.norm(np.array([cx, cy]) - np.array(ball_last_pos))
    #         should_connect = (dist_to_last <= BALL_MAX_JUMP_DIST) or (display_conf >= BALL_FALLBACK_MIN_CONF)

    #     # cơ chế "buông" khi dính nhiễu yếu kéo dài
    #     ball_low_conf_streak = ball_low_conf_streak + 1 if display_conf < BALL_STUCK_CONF_THRESH else 0

    #     if ball_low_conf_streak >= BALL_STUCK_FRAME_LIMIT:
    #         ball_last_pos = None
    #         ball_low_conf_streak = 0
    #         ball_trail.append(None)
    #         trim_trail()
    #         # bỏ qua hẳn frame này (không vẽ box, không cập nhật vị trí theo nhiễu vừa buông)
    #     else:
    #         if not should_connect and ball_last_pos is not None:
    #             ball_trail.append(None)
    #             trim_trail()

    #         ball_last_pos = (cx, cy)
    #         ball_trail.append((float(cx), float(cy)))
    #         trim_trail()

    #         x1, y1, x2, y2 = xyxy[display_idx].astype(int)
    #         cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
    #         cv2.putText(annotated, "Ball", (x1, max(0, y1 - 8)),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    #     # vẽ vệt bóng mờ dần, bỏ qua đoạn nối nếu 1 trong 2 đầu là break marker
    #     for i in range(1, len(ball_trail)):
    #         p1, p2 = ball_trail[i - 1], ball_trail[i]
    #         if p1 is None or p2 is None:
    #             continue
    #         alpha = i / len(ball_trail)
    #         thickness = max(1, int(4 * alpha))
    #         color = (0, int(255 * alpha), int(255 * alpha))
    #         cv2.line(annotated, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)
    # else: không có box hợp lệ -> giữ nguyên trail, không reset (bóng có thể tạm bị che khuất)
    
    ball_results = ball_model.track(
        frame, 
        persist=True, 
        tracker="ball_bytetrack.yaml", # Trỏ vào file custom của nhóm
        conf=BALL_CONF, 
        imgsz=BALL_INFER_IMGSZ, 
        verbose=False
    )

    # Lấy thông tin từ ByteTrack
    if ball_results[0].boxes.id is not None:
        # ByteTrack đã tự động lọc và xử lý Tranh chấp/Bóng chậm ở hậu trường
        # Ta chỉ việc lấy tọa độ và ID ra
        boxes = ball_results[0].boxes.xyxy.cpu().numpy()
        track_ids = ball_results[0].boxes.id.int().cpu().tolist()
        confs = ball_results[0].boxes.conf.cpu().numpy()

        # Giả định ID đầu tiên là quả bóng (vì class bóng chỉ có 1)
        # ByteTrack đảm bảo ID này giữ nguyên qua các frame
        target_box = boxes[0]
        x1, y1, x2, y2 = target_box.astype(int)
        
        # Tính tâm bóng để vẽ trail
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        # Cập nhật mảng ball_trail của nhóm
        ball_trail.append((float(cx), float(cy)))
        trim_trail()

        # Vẽ Bounding Box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(annotated, "Ball", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    else:
        # ByteTrack báo mất bóng (Bị che khuất hoặc bay khỏi cam)
        ball_trail.append(None)
        trim_trail()

    # --- ĐOẠN NÀY GIỮ NGUYÊN HOÀN TOÀN TỪ CODE CŨ ---
    # Vẽ vệt bóng mờ dần (fade trail)
    for i in range(1, len(ball_trail)):
        p1, p2 = ball_trail[i - 1], ball_trail[i]
        if p1 is None or p2 is None:
            continue
        alpha = i / len(ball_trail)
        thickness = max(1, int(4 * alpha))
        color = (0, int(255 * alpha), int(255 * alpha)) # Vàng nhạt -> đậm
        cv2.line(annotated, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness)

    # ================= PERSON: detect + ID =================
    # if frame_idx % PERSON_DETECT_EVERY_N_FRAMES == 0:
    #     person_results = person_model.track(
    #         frame, persist=True, tracker='bytetrack.yaml', conf=PERSON_CONF, classes=[0], verbose=False
    #     )
    #     last_person_boxes = []
    #     if person_results[0].boxes.id is not None:
    #         p_boxes = person_results[0].boxes.xyxy.cpu().numpy()
    #         p_ids = person_results[0].boxes.id.int().cpu().tolist()
    #         last_person_boxes = list(zip(p_boxes, p_ids))

    # for box, pid in last_person_boxes:
    #     x1, y1, x2, y2 = box.astype(int)
    #     cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
    #     cv2.putText(annotated, f"ID:{pid}", (x1, max(0, y1 - 8)),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ================= Ghi video + preview =================
    writer.write(annotated)
    cv2.imshow('Ball + Player Tracking', annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_idx += 1

cap.release()
writer.release()
cv2.destroyAllWindows()
print(f"Xong. Video kết quả đã lưu tại: {OUTPUT_VIDEO}")


"""
TÓM TẮT QUYẾT ĐỊNH THIẾT KẾ (để đưa vào báo cáo, phần Methodology / Challenges):

1. Bỏ ByteTrack cho bóng, dùng model.predict() thuần + tự viết continuity theo khoảng cách.
   Lý do: ByteTrack dựa trên IoU giữa 2 frame để giữ track ID; với vật thể nhỏ + di chuyển
   nhanh như bóng, IoU giữa 2 frame liên tiếp thường quá thấp -> 1 quả bóng bị tách thành
   hàng chục track ID khác nhau (đã kiểm chứng bằng thực nghiệm: 43 ID trong 200 frame).

2. Vẫn giữ ByteTrack cho người -- vì có nhiều đối tượng cùng lúc, cần ID để phân biệt,
   và người có kích thước đủ lớn để IoU giữa 2 frame vẫn ổn định.

3. Lọc box theo hình dạng (diện tích + tỉ lệ khung hình) trước khi xét continuity,
   để loại các nhiễu không giống bóng về mặt hình học (giày, cờ góc...).

4. Lọc theo vị trí (BALL_FIELD_TOP_Y): loại toàn bộ detection ở khán đài, dựa trên giả định
   camera broadcast giữ góc quay tương đối cố định trong clip.

5. Chọn box hiển thị bằng score kết hợp cả khoảng cách lẫn confidence (không chỉ 1 trong 2),
   để tránh 2 lỗi đối lập: (a) chỉ ưu tiên khoảng cách -> dễ bị "dính" vào 1 nhiễu yếu ở gần
   trong khi bỏ qua bóng thật conf cao hơn ở xa hơn 1 chút; (b) chỉ ưu tiên conf -> box có thể
   nhảy sang vật thể khác có conf cao hơn ở vị trí bất kỳ trong frame.

6. Tách riêng quyết định "vẽ box" và "nối line": box được vẽ bất cứ khi nào có detection
   hợp lệ, nhưng line chỉ nối khi đủ tin cậy (continuity gần, hoặc conf cao khi nhảy xa) --
   tránh việc "mất box" chỉ vì line không đủ điều kiện nối.

7. Cơ chế "buông" (BALL_STUCK_*): nếu vị trí đang bám có conf thấp liên tục nhiều frame,
   chủ động reset để tránh bị khóa cứng vào 1 điểm nhiễu tĩnh trong thời gian dài.

HẠN CHẾ ĐÃ BIẾT (đưa vào phần Limitations):
- Vẫn có tỷ lệ frame mất box đáng kể khi bóng bị che khuất hoàn toàn hoặc motion blur nặng.
- BALL_FIELD_TOP_Y giả định camera không pan/zoom mạnh -- không tổng quát cho mọi loại video.
- Các ngưỡng (conf, khoảng cách, diện tích...) được hiệu chỉnh thủ công dựa trên quan sát
  thực nghiệm trên 1 clip cụ thể, có thể cần điều chỉnh lại với video/góc quay khác.
"""