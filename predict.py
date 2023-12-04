from ultralytics import YOLO
import cv2
import os

# img_path = "./test-photo/all-honors.png"
# img_path = "./test-photo/mj1.png"
# img_path = "./test-photo/mj2.jpeg"
# img_path = "./test-photo/mj3.png"
# img_path = "./test-photo/mj4.jpg"

  
# model = YOLO("./runs/detect/home-mj-v1/weights/best.pt")
# model = YOLO("./runs/detect/nov2023/weights/best.pt")
model = YOLO("./runs/detect/manual-aug-56940/weights/best.pt") 

image_dir = './datasets/test/images'

for filename in os.listdir(image_dir):
    if not filename.endswith('.jpg'):
        continue

    res = model(f"{image_dir}/{filename}")
    res_plotted = res[0].plot(conf=True)
    cv2.imshow("result", res_plotted)
    key = cv2.waitKey(0)
