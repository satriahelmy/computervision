import base64

import cv2
import numpy as np
import torch
from flask import Blueprint, render_template, request

from module import allowed_file

bp = Blueprint("custom_object_detection", __name__)

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model = torch.hub.load("ultralytics/yolov5",
                       "custom",
                       path="./models/best.pt",
                       force_reload=True)


@bp.route("/custom_object_detection", methods=["GET", "POST"])
def custom_object_detection():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file and allowed_file(image_file.filename):
            image_data = image_file.read()
            image_np = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

            # Perform object detection
            detection_result = model(image)
            print(detection_result)

            # Get YOLOv5's default class labels
            class_labels = detection_result.names

            for det in detection_result.pred[0]:
                x1, y1, x2, y2, conf, cls = det.tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                label = class_labels[int(cls)]  # Get the class label
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, text=label, org=(x1, y1 - 10), fontFace=cv2.FONT_HERSHEY_DUPLEX,
                            fontScale=0.5, color=(0, 255, 0), thickness=2)

            # Encode the detected image in base64
            _, buffer = cv2.imencode('.jpg', image)
            image_base64 = base64.b64encode(buffer).decode()
            return render_template("pages/custom_object_detection.html",
                                   result=image_base64,
                                   class_labels=label)

    # Handle GET request
    return render_template("pages/custom_object_detection.html")
