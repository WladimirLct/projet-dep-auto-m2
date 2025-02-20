from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("yolo11n.pt")

model.export(format="onnx", dynamic=True, nms=True)

import os

os.remove("yolo11n.pt")
os.rename("yolo11n.onnx", "model/yolo11.onnx")