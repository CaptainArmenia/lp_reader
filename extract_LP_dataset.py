from collections import defaultdict
import argparse
import os
from glob import glob

from tqdm import tqdm
import cv2
import numpy as np

from ultralytics import YOLO
from read import Reader


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', type=str, help='Directory to read videos from')
parser.add_argument('--output_dir', type=str, help='Directory to save the output to')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Count files in output directory
output_files = glob(os.path.join(args.output_dir, "*.jpg"))
output_count = len(output_files)

# Load a pretrained YOLOv8n model
model = YOLO('runs/detect/train3/weights/best.pt')

# Open the video file
video_paths = glob(os.path.join(args.video_dir, "*.mp4"))
video_paths.sort()

print(f"Found {len(video_paths)} videos")

for video_path in video_paths:
    basename = video_path.split('/')[-1]
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Store the track history
    track_history = defaultdict(lambda: [])
    stop_counter = defaultdict(lambda: 0)
    saved_tracks = []

    # Loop through the video frames
    for i in tqdm(range(frame_count)):
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, verbose=False)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]

                    if track_id not in saved_tracks:

                        # Check if object has been relatively quiet for the last 15 frames and bounding box area is sufficiently large
                        if len(track) > 15:
                            # Get the last 15 points
                            last_15_points = track[-15:]
                            # Calculate the average distance between the last 15 points
                            distances = np.linalg.norm(np.diff(last_15_points, axis=0), axis=1)
                            average_distance = np.mean(distances)
                            # If the average distance is less than 10 pixels and box area is sufficiently large, start a stop counter
                            if average_distance < 3 and stop_counter[track_id] < 30 and (w * h) > 4000:
                                stop_counter[track_id] += 1
                            else:
                                stop_counter[track_id] = 0
                                
                        track.append((float(x), float(y)))  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)

                        # If the stop counter reaches 30, save the license plate crop
                        if stop_counter[track_id] == 30 and track_id not in saved_tracks:
                            license_plate_crop = frame[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]
                            cv2.imwrite(os.path.join(args.output_dir, f"{str(output_count).zfill(6)}.jpg"), license_plate_crop)
                            stop_counter[track_id] += 1
                            output_count += 1
                            saved_tracks.append(track_id)
               
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
