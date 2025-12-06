from ultralytics import YOLO
from .yolo.inference.yolo_model import Yolo_Detection
from .resnet.inference.resnet_model import Resnet_Classifier
from PIL import Image
import io
import cv2
import numpy as np

class Yolo_Stacked:
    #store important model things
    def __init__(self, resnet_path):
        self.resnet = Resnet_Classifier(resnet_path) 

    #first call the yolo class's predict to generate all bounding boxes
    #then use check bounding box as input to dector and output prediction
    def predict(self, yolo):
        signatures = yolo.signatures_last_prediction()[0]
        if not signatures:
            raise ValueError("No signature crops found in last YOLO prediction.")
        # self.resnet.visualize_model_predictions(signatures)
        preds, conf, probs = self.resnet.predict(signatures)
        return preds, conf, probs
