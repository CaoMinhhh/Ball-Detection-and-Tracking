import os
from ultralytics import YOLO

def main():
    # --- 1. MODEL INITIALIZATION ---
    # Use 'yolov8n.pt' for a fresh training run. 
    # If resuming an interrupted training, change this to your specific checkpoint (e.g., 'runs/detect/train/weights/last.pt')
    weights_path = 'yolov8n.pt' 
    
    print(f"[*] Initializing YOLOv8 model with weights: {weights_path}")
    model = YOLO(weights_path)

    # --- 2. START TRAINING PIPELINE ---
    print("[>] Starting the training process...")
    results = model.train(
        data='soccer.yaml',      # Dataset configuration file
        epochs=50,               # Total number of training epochs
        imgsz=960,               # Input image size (optimized for small ball detection)
        batch=10,                # Batch size (adjusted based on available GPU VRAM, e.g., 6GB+)
        workers=6,               # Number of dataloader workers (tune based on available system RAM)
        device=0,                # GPU device ID (use 'cpu' if no GPU is available)
        patience=15,             # Early stopping patience (stop if no improvement for 15 epochs)
        cache=False,             # Disable RAM/Disk caching to prevent out-of-memory issues
        plots=False,             # Disable plots to reduce disk I/O overhead during each epoch
        amp=True,                # Enable Automatic Mixed Precision (AMP) for faster training
        val=True,                # Enable validation at the end of each epoch
    )
    print("[+] Training completed successfully!")

if __name__ == '__main__':
    # Required for Windows multiprocessing compatibility
    main()