from ultralytics import YOLO
from PIL import Image
import numpy as np

class Yolo_Detection:
    def __init__(self, path):
        self.weights = path
        self.model = YOLO(self.weights)
        self.result = [] 
        self.results = []
    
    def predict(self, img):
        upload = Image.open(img).convert("RGB")
        np_img = np.array(upload)
        self.result = self.model(np_img)
        self.results.append(self.result[0])

    #returns np image 
    def visualize_prediction(self):
        if self.result:
            annotated = self.result[0].plot()
            annotated = annotated[:, :, ::-1]
            return annotated

    def visualize_all_bounding(self):
        return [result[0].plot()[:, :, ::-1] for result in self.results] 