import base64

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from yolo.core import ONNXYOLODetector

# Load the YOLO ONNX model
model_path = "model/yolo11.onnx"
yolov11_detector = ONNXYOLODetector(
    model_path=model_path, conf_thresh=0.1, iou_thresh=0.45
)

# Create the FastAPI app
app = FastAPI(title="YOLO ONNX Inference API")


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    """
    Receives an image file, runs YOLO inference, draws the detections on the image,
    encodes the annotated image as JPEG, and returns it as a base64 string.
    """
    # Read the image from the uploaded file
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Detect objects in the image
    boxes, scores, class_ids = yolov11_detector.detect(image)
    frame = yolov11_detector.draw_detections(image, boxes, scores, class_ids)

    # Encode the annotated image as JPEG
    ret, buffer = cv2.imencode(".jpg", frame)
    if not ret:
        return {"error": "Failed to encode image"}

    # Convert the JPEG to a base64 string
    encoded_image = base64.b64encode(buffer).decode("utf-8")
    return {"annotated_image": encoded_image}
