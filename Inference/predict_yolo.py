import numpy as np
import cv2
from ultralytics import YOLO
import streamlit as st


@st.cache_resource
def load_yolo_model(weights_path: str):
    """Loads YOLO model once."""
    return YOLO(weights_path)


def run_yolo_inference(image_bytes, weights_path):
    """
    Runs YOLO inference and returns:
        - annotated image (numpy array)
        - detection list
    """
    model = load_yolo_model(weights_path)

    # Convert uploaded bytes → CV2 image
    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Run prediction
    results = model(img)

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

    return annotated, detections
