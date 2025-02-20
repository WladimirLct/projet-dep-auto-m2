# Author: Sihab Sahariar
# Date: 2024-10-21
# License: MIT License
# Email: sihabsahariarcse@gmail.com

import argparse
import os
import sys
import os.path as osp
import cv2
import numpy as np
import onnxruntime as ort
from math import exp

# Constants and configurations
CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
           'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
           'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
           'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
           'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
           'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
           'hair drier', 'toothbrush']


COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

meshgrid = []
class_num = len(CLASSES)
headNum = 3
strides = [8, 16, 32]
mapSize = [[80, 80], [40, 40], [20, 20]]
input_imgH = 640
input_imgW = 640


class DetectBox:
    def __init__(self, classId, score, xmin, ymin, xmax, ymax):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


class ONNXYOLODetector:
    def __init__(self, model_path='./yolov11n.onnx', conf_thresh=0.5, iou_thresh=0.45):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.ort_session = ort.InferenceSession(self.model_path)
        self.generate_meshgrid()

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    @staticmethod
    def preprocess_image(img_src, resize_w, resize_h):
        image = cv2.resize(img_src, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        image /= 255.0
        return image

    def generate_meshgrid(self):
        for index in range(headNum):
            for i in range(mapSize[index][0]):
                for j in range(mapSize[index][1]):
                    meshgrid.append(j + 0.5)
                    meshgrid.append(i + 0.5)

    def iou(self, xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
        xmin = max(xmin1, xmin2)
        ymin = max(ymin1, ymin2)
        xmax = min(xmax1, xmax2)
        ymax = min(ymax1, ymax2)

        innerWidth = max(0, xmax - xmin)
        innerHeight = max(0, ymax - ymin)

        innerArea = innerWidth * innerHeight
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        total = area1 + area2 - innerArea

        return innerArea / total

    def nms(self, detectResult):
        predBoxs = []
        sort_detectboxs = sorted(detectResult, key=lambda x: x.score, reverse=True)

        for i in range(len(sort_detectboxs)):
            if sort_detectboxs[i].classId != -1:
                predBoxs.append(sort_detectboxs[i])
                for j in range(i + 1, len(sort_detectboxs), 1):
                    if sort_detectboxs[i].classId == sort_detectboxs[j].classId:
                        iou = self.iou(
                            sort_detectboxs[i].xmin, sort_detectboxs[i].ymin,
                            sort_detectboxs[i].xmax, sort_detectboxs[i].ymax,
                            sort_detectboxs[j].xmin, sort_detectboxs[j].ymin,
                            sort_detectboxs[j].xmax, sort_detectboxs[j].ymax
                        )
                        if iou > self.iou_thresh:
                            sort_detectboxs[j].classId = -1
        return predBoxs

    def postprocess(self, predictions, img_h, img_w):
        """
        Process detections from the (300,6) tensor.
        Assumes each detection is in the order:
        [x1, y1, x2, y2, confidence, class_id]
        and the coordinates are relative to the resized image (640x640).
        """
        detectResult = []
        scale_w = img_w / input_imgW  # input_imgW is 640
        scale_h = img_h / input_imgH  # input_imgH is 640
        print(len(predictions[0]))

        for det in predictions:
            x1, y1, x2, y2, conf, class_id = det

            if conf > self.conf_thresh:
                # Scale the box coordinates back to the original image size.
                new_x1 = max(0, x1 * scale_w)
                new_y1 = max(0, y1 * scale_h)
                new_x2 = min(img_w, x2 * scale_w)
                new_y2 = min(img_h, y2 * scale_h)

                box = DetectBox(int(class_id), conf, new_x1, new_y1, new_x2, new_y2)
                detectResult.append(box)

        # Apply non-maximum suppression to filter overlapping boxes.
        detectResult = self.nms(detectResult)
        return detectResult

    def detect(self, img_path):
        if isinstance(img_path, str):
            orig = cv2.imread(img_path)
        else:
            orig = img_path

        img_h, img_w = orig.shape[:2]
        image = self.preprocess_image(orig, input_imgW, input_imgH)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)

        # Use the proper input key if needed. (For example, change 'data' to the expected input name.)
        input_name = self.ort_session.get_inputs()[0].name
        pred_results = self.ort_session.run(None, {input_name: image})

        prediction = pred_results[0][0]  # shape: (300, 6)
        predbox = self.postprocess(prediction, img_h, img_w)

        boxes = []
        scores = []
        class_ids = []

        for box in predbox:
            boxes.append([int(box.xmin), int(box.ymin), int(box.xmax), int(box.ymax)])
            scores.append(box.score)
            class_ids.append(box.classId)

        return boxes, scores, class_ids



    def draw_detections(self,image, boxes, scores, class_ids, mask_alpha=0.3):
        """
        Combines drawing masks, boxes, and text annotations on detected objects.
        
        Parameters:
        - image: Input image.
        - boxes: Array of bounding boxes.
        - scores: Confidence scores for each detected object.
        - class_ids: Detected object class IDs.
        - mask_alpha: Transparency of the mask overlay.
        """
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        mask_img = image.copy()

        # Draw bounding boxes, masks, and text annotations
        for class_id, box, score in zip(class_ids, boxes, scores):
            color = COLORS[class_id]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            # Draw fill rectangle for mask
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)
            
            # Draw bounding box
            cv2.rectangle(det_img, (x1, y1), (x2, y2), color, 2)

            # Prepare text (label and score)
            label = CLASSES[class_id]
            caption = f'{label} {int(score * 100)}%'
            
            # Calculate text size and position
            (tw, th), _ = cv2.getTextSize(text=caption, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                        fontScale=font_size, thickness=text_thickness)
            th = int(th * 1.2)
            
            # Draw filled rectangle for text background
            cv2.rectangle(det_img, (x1, y1), (x1 + tw, y1 - th), color, -1)
            
            # Draw text over the filled rectangle
            cv2.putText(det_img, caption, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size,
                        (255, 255, 255), text_thickness, cv2.LINE_AA)

        # Blend the mask image with the original image
        det_img = cv2.addWeighted(mask_img, mask_alpha, det_img, 1 - mask_alpha, 0)

        return det_img