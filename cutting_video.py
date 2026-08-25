import os
import cv2

# --- 1. CONFIGURATION ---
INPUT_VIDEO = "full_match.mp4"   # The raw downloaded video (e.g., via yt-dlp)
OUTPUT_VIDEO = "clip_test.mp4"   # The output filename

# Specify the start time for the clip
START_MIN = 7
START_SEC = 42

# Recommended duration: ~90 seconds (1m30s) for smooth testing and debugging
DURATION_SEC = 90


def main():
    print("[*] Initializing video cutter...")

    # --- 2. INPUT VALIDATION ---
    if not os.path.exists(INPUT_VIDEO):
        print(f"[!] Error: Input video '{INPUT_VIDEO}' not found!")
        print("[i] Please download a full match video and place it in the same directory to use this script.")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print(f"[!] Error: Could not open '{INPUT_VIDEO}'. The container/codec may not be supported by OpenCV.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print(f"[!] Error: Could not read FPS. The video file '{INPUT_VIDEO}' might be corrupted.")
        cap.release()
        return

    # --- 3. TIME TO FRAME CALCULATION ---
    start_time_sec = (START_MIN * 60) + START_SEC
    start_frame = int(start_time_sec * fps)
    end_frame = int((start_time_sec + DURATION_SEC) * fps)

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_video_frames > 0 and start_frame >= total_video_frames:
        print(f"[!] Error: Requested start time ({START_MIN:02d}:{START_SEC:02d}) is beyond the video length "
              f"({total_video_frames} frames, ~{total_video_frames / fps:.1f}s).")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[*] Target video: {INPUT_VIDEO} ({fps:.2f} FPS, {w}x{h})")
    print(f"[*] Cutting from {START_MIN:02d}:{START_SEC:02d} for {DURATION_SEC} seconds...")

    # Using OpenCV's built-in codec (mp4v). No external system FFmpeg required!
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (w, h))

    # --- 4. VIDEO PROCESSING LOOP ---
    frame_idx = start_frame
    total_frames = end_frame - start_frame
    processed_frames = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            print("\n[!] Warning: Reached the end of the video before finishing the requested duration.")
            break

        out.write(frame)

        frame_idx += 1
        processed_frames += 1

        # Simple progress indicator (updates every 10 seconds of video content)
        if processed_frames % int(fps * 10) == 0:
            print(f"    -> Processed {processed_frames}/{total_frames} frames...")

    cap.release()
    out.release()

    if processed_frames == 0:
        print(f"\n[!] Warning: No frames were written. '{OUTPUT_VIDEO}' may be empty or invalid.")
    else:
        print(f"\n[+] DONE! Clip saved successfully as '{OUTPUT_VIDEO}' ({processed_frames} frames written).")


if __name__ == '__main__':
    main()