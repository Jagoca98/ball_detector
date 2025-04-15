import cv2
import numpy as np
import os
import torch
from tqdm import tqdm
from collections import defaultdict

from yolox.tracker.byte_tracker import BYTETracker

# Configs
IMG_DIR = "/bytetrack/datasets/eris_2/camera_front"
DET_DIR = "/bytetrack/datasets/eris_2/camera_front_detections"
OUTPUT_DIR = "/data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

label_map = {}
label_id_counter = [0]

def get_label_id(label):
    if label not in label_map:
        label_map[label] = label_id_counter[0]
        label_id_counter[0] += 1
    return label_map[label]

class Args:
    track_thresh = 0.5
    track_buffer = 30
    match_thresh = 0.8
    aspect_ratio_thresh = 9999  # ← prevents rejection of wide boxes
    min_box_area = 0            # ← allows even small boxes
    mot20 = False

tracker = BYTETracker(Args(), frame_rate=30)

# Read all image filenames
img_filenames = sorted([
    f for f in os.listdir(IMG_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))
])

def parse_detection(line):
    try:
        # Split only on first two spaces (label + score)
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            raise ValueError("Detection line does not contain label, score, and coords.")

        label = parts[0]
        score = float(parts[1])

        # Normalize coordinates: remove brackets and commas
        coords_raw = parts[2].strip('[]').replace(',', ' ')
        coords = [int(val) for val in coords_raw.strip().split() if val.isdigit() or (val[0] == '-' and val[1:].isdigit())]

        if len(coords) != 8:
            raise ValueError(f"Expected 8 integers for 4 (x,y) points, got {len(coords)} → {coords}")

        points = np.array(coords).reshape(-1, 2)

        x_min = points[:, 0].min()
        y_min = points[:, 1].min()
        x_max = points[:, 0].max()
        y_max = points[:, 1].max()

        x = x_min
        y = y_min
        w = x_max - x_min
        h = y_max - y_min

        class_id = label_map.get(label, 0)  # fallback to 0 if unknown label
        return [x, y, x_max, y_max, score, class_id], points

    except Exception as e:
        print(f"[⚠️ Skipped] {line.strip()}\n  ↳ Error: {e}")
        return None


def get_detections_for_frame(frame_name):
    det_path = os.path.join(DET_DIR, os.path.splitext(frame_name)[0] + ".txt")
    if not os.path.exists(det_path):
        return []
    with open(det_path, "r") as f:
        lines = f.readlines()
    return [parse_detection(line) for line in lines]

# Create a video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 10
video_size = (1920, 1080)  # Change this to your video size
video_writer = cv2.VideoWriter(os.path.join(OUTPUT_DIR, "output.avi"), fourcc, fps, video_size)

# Main loop over frames
for frame_name in tqdm(img_filenames, desc="Processing frames", unit="frame"):
    frame_path = os.path.join(IMG_DIR, frame_name)
    frame = cv2.imread(frame_path)
    if frame is None:
        continue

    h, w = frame.shape[:2]
    parsed = [p for p in get_detections_for_frame(frame_name) if p is not None]
    detections = np.array([p[0] for p in parsed]) if parsed else np.empty((0, 6))
    detections = torch.tensor(detections, dtype=torch.float32)
    polygons = [p[1] for p in parsed]

    online_targets = tracker.update(detections, img_info=(h, w), img_size=(h, w))

    # Draw tracked bounding boxes
    for t in online_targets:
        x, y, w_box, h_box = t.tlwh
        tid = t.track_id
        cv2.rectangle(frame, (int(x), int(y)), (int(x + w_box), int(y + h_box)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {tid}', (int(x), int(y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
    # # Draw polygons
    # for polygon in polygons:
    #     cv2.polylines(frame, [polygon], isClosed=True, color=(0, 0, 255), thickness=2)

    # Resize frame to match video size
    resized_frame = cv2.resize(frame, video_size)
    video_writer.write(resized_frame)

# Release the video writer
video_writer.release()

print("✅ All images processed and saved to 'output/'")
