import numpy as np
import cv2
import io
import streamlit as st
from ultralytics import YOLO
from Yolo_Stacked.yolo.inference.yolo_model import Yolo_Detection
from Yolo_Stacked.yolo_stacked import Yolo_Stacked



@st.cache_resource
def load_yolo_model(weights_path: str):
    """Loads YOLO model once."""
    return Yolo_Detection(weights_path)

@st.cache_resource
def load_resnet_model(weights_path: str):
    """Loads YOLO model once."""
    return Yolo_Stacked(weights_path)


def run_yolo_stacked_inference(image_bytes, yolo_weights_path, resnet_weights_path):
    """
    Runs YOLO inference and returns:
        - annotated image (numpy array)
        - detection list
    """
    yolo = load_yolo_model(yolo_weights_path)
    stacked = load_resnet_model(resnet_weights_path)

    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # ----- YOLO PREDICTION -----

    # Run prediction
    results = yolo.predict(img)

    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = results[0].names[cls]
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append(
            {
                "label": label,
                "confidence": round(conf, 3),
                "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            }
        )

    annotated = results[0].plot()  # numpy image with boxes

    # ----- RESNET PREDICTION -----
    preds, conf, probs = stacked.predict(yolo)

    # print("preds", preds)
    # print("conf", conf)
    # print("prob", probs)

    return preds, annotated, detections