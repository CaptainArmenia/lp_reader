from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.yaml').load('yolov8s.pt')  # build from YAML and transfer weights

# Train the model
results = model.train(data='/home/andy/Desktop/datasets/LPR/dataset.yaml', epochs=100, imgsz=1280, batch=8)