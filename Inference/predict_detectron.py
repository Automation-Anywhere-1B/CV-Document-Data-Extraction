import numpy as np
import cv2
import streamlit as st

from Detectron2.predict import get_trained_predictor

THRESH = 0.40


@st.cache_resource
def load_detectron_model():
    predictor, class_names = get_trained_predictor()
    return predictor, class_names


def run_detectron_inference(image_bytes):
    """
    Runs Detectron2 inference on a single uploaded image and returns:
        - annotated image (numpy array with boxes)
        - detection list [{label, confidence, bbox}, ...]
    """
    predictor, class_names = load_detectron_model()

    file_bytes = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    outputs = predictor(img)
    inst = outputs["instances"].to("cpu")
    boxes = inst.pred_boxes.tensor.numpy().tolist()
    scores = inst.scores.numpy().tolist()
    classes = inst.pred_classes.numpy().tolist()

    detections = []
    for (box, score, cls) in zip(boxes, scores, classes):
        if score < THRESH:
            continue

        x1, y1, x2, y2 = map(int, box)
        label = class_names[cls] if 0 <= cls < len(class_names) else f"id{cls}"

        detections.append(
            {
                "label": label,
                "confidence": round(float(score), 3),
                "bbox": [x1, y1, x2, y2],
            }
        )

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{label} {int(score * 100)}%"

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(
            img,
            (x1, y1 - th - 6),
            (x1 + tw + 4, y1),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            img,
            text,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return img, detections