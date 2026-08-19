import cv2

cap = cv2.VideoCapture("full_match.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)

start_sec = 7 * 60 + 42   # 7:42 - thời điểm bạn muốn bắt đầu cắt
duration_sec = 90          # cắt 90 giây

start_frame = int(start_sec * fps)
end_frame = int((start_sec + duration_sec) * fps)

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # codec built-in của OpenCV, không cần ffmpeg hệ thống
out = cv2.VideoWriter("clip_test.mp4", fourcc, fps, (w, h))

frame_idx = start_frame
while frame_idx < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
print("Xong, đã lưu clip_test.mp4")