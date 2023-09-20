import os
from glob import glob

from tqdm import tqdm
import cv2
import numpy as np

from detect import LPDetector


# Use the LPDetector class to detect license plates in a video and save the frames containing a license plate
# to a folder.
def split_frames(source, destination, detector, global_count=0):
    cap = cv2.VideoCapture(source)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count)):
        success, frame = cap.read()
        if success:
            if i % 10 != 0:
                continue
            results = detector.detect(frame)
            if len(results[0].boxes) > 0:
                # Save each bounding box to a separate file
                for j, box in enumerate(results[0].boxes):
                    x1, y1, x2, y2 = np.array(box.xyxy[0].cpu()).astype(int)
                    # check if the bounding box is too small
                    if (x2 - x1) *(y2 - y1) < 4000:
                        continue
                    cropped = frame[y1:y2, x1:x2]
                    cv2.imwrite(os.path.join(destination, f"{i + global_count}_{j}.jpg"), cropped)
                global_count += 1
        else:
            break
    cap.release()
    return frame_count + global_count

def process_folder(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    detector = LPDetector()
    global_count = 0
    for video in glob(os.path.join(source, "*.mp4")):
        print(f"Processing {os.path.basename(video)}")
        global_count = split_frames(video, destination, detector, global_count)

if __name__ == '__main__':
    source = '/home/andy/Desktop/datasets/lo_valledor'
    destination = '/home/andy/Desktop/datasets/lo_valledor_frames'
    process_folder(source, destination)