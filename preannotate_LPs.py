from collections import defaultdict
import argparse
import os
from glob import glob

from tqdm import tqdm
import cv2
import numpy as np

from read import Reader


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--lp_dir', type=str, help='Directory to read license plates from')
parser.add_argument('--output_dir', type=str, help='Directory to save annotated license plates to')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

# Load OCR model
reader = Reader()

lp_files = glob(os.path.join(args.lp_dir, "*.jpg"))
lp_files.sort()
print(f"Found {len(lp_files)} license plates")

# save classes file
with open(os.path.join(args.output_dir, "classes.txt"), "w") as f:
    for c in reader.get_classes():
        f.write(f"{c}\n")

# Read each license plate image and save detections as YOLO annotations
for lp_file in tqdm(lp_files):
    basename = lp_file.split('/')[-1]

    # Read the license plate image
    img = cv2.imread(lp_file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Run OCR on the license plate
    xmin, ymin, xmax, ymax, label_ids, conf = reader.detect(img)
    
    # Transform coordinates to YOLO format
    xmin = np.array(xmin) / img.shape[1]
    xmax = np.array(xmax) / img.shape[1]
    ymin = np.array(ymin) / img.shape[0]
    ymax = np.array(ymax) / img.shape[0]

    x = xmin
    y = ymin
    width = xmax - xmin
    height = ymax - ymin

    # transform top-left coordinates to center coordinates
    x = x + width / 2
    y = y + height / 2

    # sort detections by x coordinate
    sort_idx = np.argsort(x)
    x = x[sort_idx]
    y = y[sort_idx]
    width = width[sort_idx]
    height = height[sort_idx]
    label_ids = np.array(label_ids)[sort_idx]


    # Save annotations
    with open(os.path.join(args.output_dir, basename.replace(".jpg", ".txt")), "w") as f:
        for i in range(len(xmin)):
            # for each box, write class_id, x, y, width, height with 6 decimals
            f.write(f"{label_ids[i]} {x[i]:.6f} {y[i]:.6f} {width[i]:.6f} {height[i]:.6f}\n")


