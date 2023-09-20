from collections import defaultdict
import argparse
import os
from glob import glob

import cv2
import numpy as np

from ultralytics import YOLO
from read import Reader


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--save_video', action='store_true', help='Save the video to a file')
parser.add_argument('--video_dir', type=str, help='Directory to read videos from')
parser.add_argument('--video_file', type=str, help='Name of the video to read')
args = parser.parse_args()

save_video = args.save_video

# Load a pretrained YOLOv8n model
model = YOLO('weights/lp/best.pt')

# Load OCR model
reader = Reader()

# Open the video file
video_paths = glob(os.path.join(args.video_dir, "*.mp4"))
video_paths.sort()

print(f"Found {len(video_paths)} videos")

for video_path in video_paths:
    basename = video_path.split('/')[-1]

    cap = cv2.VideoCapture(video_path)

    if save_video:
        # cv2 video writer for mp4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(basename, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

    # Store the track history
    track_history = defaultdict(lambda: [])
    stop_counter = defaultdict(lambda: 0)
    license_plates = defaultdict(lambda: [])
    license_plates_confidences = defaultdict(lambda: [])

    # Count vehicles
    registered_license_plates = []

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        vis_frame = frame.copy()

        if success:
            # Run YOLOv8 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, verbose=False)

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            if results[0].boxes.id is not None:
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]

                    # Check if object has been relatively quiet for the last 15 frames and bounding box area is sufficiently large
                    if len(track) > 15:
                        # Get the last 15 points
                        last_15_points = track[-15:]
                        # Calculate the average distance between the last 15 points
                        distances = np.linalg.norm(np.diff(last_15_points, axis=0), axis=1)
                        average_distance = np.mean(distances)
                        
                        # If the average distance is less than 10 pixels and box area is sufficiently large, start a stop counter
                        if average_distance < 3 and w * h > 3000:
                            stop_counter[track_id] += 1
                        else:
                            stop_counter[track_id] = 0
                            
                    track.append((float(x), float(y)))  # x, y center point

                    if len(track) > 30:  # retain 90 tracks for 90 frames
                        track.pop(0)

                    # If the stop counter reaches 30, read and save the license plate
                    if stop_counter[track_id] > 29 and stop_counter[track_id] < 40:
                        license_plate_crop = frame[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]

                        # check if crop is not blurry
                        if cv2.Laplacian(license_plate_crop, cv2.CV_64F).var() < 300:
                            stop_counter[track_id] = 0
                            continue

                        license_plate, lp_conf = reader.read(license_plate_crop)
                        
                        # Check license plate validity
                        if len(license_plate) != 6 or license_plate[0].isdigit() or license_plate[1].isdigit() or not license_plate[4].isdigit() or not license_plate[5].isdigit():
                            stop_counter[track_id] = 0
                        # Update license plate if it is valid
                        else:
                            if track_id not in license_plates:
                                license_plates[track_id] = license_plate
                                license_plates_confidences[track_id] = lp_conf
                            else:
                                if lp_conf > license_plates_confidences[track_id]:
                                    license_plates[track_id] = license_plate
                                    license_plates_confidences[track_id] = lp_conf

                            registered_license_plates = list(license_plates.values())

                            # show the license plate crop, together with its bluriness and the license plate
                            license_plate_crop_copy = license_plate_crop.copy()
                            cv2.putText(license_plate_crop_copy, "{:.2f}".format(cv2.Laplacian(license_plate_crop, cv2.CV_64F).var()), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                            cv2.imshow("License Plate", license_plate_crop_copy)
        
                     # Analyze LP only if it has not already been read
                    if track_id not in license_plates:
                        # For each pair of consecutive points, draw a line between them with a thickness proportional to the age of the point
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))

                        for j in range(1, len(points)):
                            age = int(255 * j / len(points))
                            cv2.line(vis_frame, (points[j - 1][0][0], points[j - 1][0][1]), (points[j][0][0], points[j][0][1]), (0, 255, 0), 2 + int(age / 15))

                        # For each stop counter, draw a translucid circle around its track with a diameter proportional to the counter
                        if stop_counter[track_id] > 0 and stop_counter[track_id] < 30:
                            overlay = vis_frame.copy()
                            cv2.circle(overlay, (int(x), int(y)), int(100 * (1 - 1 / stop_counter[track_id]) ** 3), (0, 255, 0), -1)
                            cv2.addWeighted(overlay, 0.5, vis_frame, 0.5, 0, vis_frame)

                    # If track has an associated license plate, draw it on the frame beneath the bounding box
                    else:
                        license_plate = license_plates[track_id]
                        # Draw a blue bounding box around the object
                        cv2.rectangle(vis_frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (255, 0, 0), 3)
                        # Write the license plate below the bounding box
                        cv2.putText(vis_frame, license_plate, (int(x - w / 2), int(y + h / 2 + 30)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        # put confidence below with two decimals
                        cv2.putText(vis_frame, "{:.2f}".format(license_plates_confidences[track_id]), (int(x - w / 2), int(y + h / 2 + 60)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # Write the number of vehicles on the bottom left corner
            #cv2.putText(frame, "Ingresos: {}".format(len(inside_vehicle_ids)), (10, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 200, 0), 2)    
            # Write the number of recognized license plates on the bottom left corner over a black rectangle
            cv2.rectangle(vis_frame, (0, vis_frame.shape[0] - 40), (vis_frame.shape[1], vis_frame.shape[0]), (0, 0, 0), -1)
            cv2.putText(vis_frame, "Patentes registradas: {}".format(len(registered_license_plates)), (10, vis_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255,255), 2)

            if save_video:
                out.write(vis_frame)
            else:
                cv2.imshow("YOLOv8 Tracking", vis_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                # skip current video if 's' is pressed
                elif cv2.waitKey(1) & 0xFF == ord("s"):
                    break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()
