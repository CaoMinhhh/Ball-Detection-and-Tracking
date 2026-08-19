from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('runs/detect/train/weights/last.pt')
    results = model.train(
        data='soccer.yaml',
        epochs=50,
        imgsz=960,
        batch=10,           # tăng vì VRAM còn dư (2.2/6GB)
        workers=6,           # cẩn thận: RAM ít, tăng workers quá cao cũng có thể ngốn RAM (mỗi worker giữ 1 phần dữ liệu buffer)
        device=0,
        patience=15,
        cache=False,         # giữ False, không dùng ram lẫn disk
        plots=False,         # giảm I/O mỗi epoch
        amp=True,
        val=True,
    )