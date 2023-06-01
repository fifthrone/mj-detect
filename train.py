from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
# model = YOLO("./runs/detect/train/weights/last.pt")
# model = YOLO("./runs/detect/train/weights/best.pt")
model = YOLO("yolov8m.pt")

# Train
results = model.train(data="data.yaml", imgsz=640, epochs=50, batch=8, name="mjv1")

# Evaluate the model's performance on the validation set
results = model.val()
