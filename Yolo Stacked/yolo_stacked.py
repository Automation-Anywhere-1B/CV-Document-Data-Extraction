from ultralytics import YOLO
from PIL import Image
import numpy as np

class Yolo_Stacked:
    #TODO
    #store important model things
    def __init__(self, path):
        self.yolo = None
        self.siamese = None

    #TODO
    #first call the yolo class's predict to generate all bounding boxes
    #then use check bounding box as input to dector and output prediction
    def predict(self, img):
        pass

    #TODO 
    #show predictions after classifying
    #returns np image 
    def visualize_prediction(self):
        pass
    
    #TODO 
    #visualize all previously made predictions
    def visualize_all_predictions(self):
        pass