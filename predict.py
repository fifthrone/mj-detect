from ultralytics import YOLO
import cv2

img_path = "./test_photo/jp_mahjong_2.jpg"

model = YOLO("./runs/detect/train/weights/best.pt")

res = model(img_path)

res_plotted = res[0].plot()
cv2.imshow("result", res_plotted)
key = cv2.waitKey(0)
