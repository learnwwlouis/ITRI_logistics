from ultralytics import YOLO

import cv2

model = YOLO("model/package_1030_n.pt")

result = model.predict(source="/home/wwlouis/project/Dataset/兆輝科技物流影片/智慧物流/05_12_19_00_44.mp4",project="result/segment/",name="package",show=False,save=True)

print(result)
