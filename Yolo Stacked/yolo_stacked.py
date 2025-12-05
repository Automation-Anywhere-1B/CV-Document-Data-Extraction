from ultralytics import YOLO
from yolo.inference.yolo_model import Yolo_Detection
from resnet.inference.resnet_model import Resnet_Classifier
from PIL import Image
import numpy as np

class Yolo_Stacked:
    #store important model things
    def __init__(self, path):
        self.yolo = Yolo_Detection(path) 
        self.resnet = Resnet_Classifier(path) 

    #first call the yolo class's predict to generate all bounding boxes
    #then use check bounding box as input to dector and output prediction
    def predict(self, img):
        self.yolo.predict('check.jpg')
        self.yolo.visualize_last_prediction()
        img = self.yolo.signatures_last_prediction()[0]
        self.resnet.visualize_model_predictions(img)