from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
# model = YOLO("./runs/detect/train/weights/last.pt")
# model = YOLO("./runs/detect/train/weights/best.pt")
# model = YOLO("./runs/detect/home-mj-v1/weights/last.pt")
model = YOLO('yolov8x.pt')

# Train
results = model.train(data="data.yaml", imgsz=640, epochs=100, batch=16, name="manual-aug-704160")

# Evaluate the model's performance on the validation set
# results = model.val()
