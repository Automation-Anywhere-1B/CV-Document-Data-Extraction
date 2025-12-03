from ultralytics import YOLO
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class Yolo_Detection:
    def __init__(self, path):
        self.weights = path
        self.model = YOLO(self.weights)
        self.predictions = []

    #returns list of Result objects and stores in array of all predictions
    def predict(self, imgs):
        if not isinstance(imgs, list):
            imgs = [imgs]
        images = [Image.open(img) for img in imgs]
        pred = self.model(images)
        self.predictions.append(pred)
        return pred

    #returns np image array
    def visualize_last_prediction(self):
        if not self.predictions:
            return
        batch = self.predictions[-1]
        imgs = [batch[i].plot() for i in range(len(batch))]
        num_imgs = len(imgs)
        cols = 3 if num_imgs >= 3 else num_imgs
        rows = num_imgs//cols if num_imgs % cols == 0 else num_imgs//cols + 1
        fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))

        axs = np.atleast_1d(axs).flatten()

        for ax, img in zip(axs, imgs):
            ax.imshow(img)
            ax.axis('off')  

        for ax in axs[len(imgs):]:
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()


    def visualize_all_predictions(self):
        if not self.predictions:
            return
        imgs = []
        num_imgs = 0
        for batch in self.predictions:
            imgs.extend([batch[i].plot() for i in range(len(batch))])
            num_imgs += len(batch)
        cols = 3 if num_imgs >= 3 else num_imgs
        rows = num_imgs//cols if num_imgs % cols == 0 else num_imgs//cols + 1
        fig, axs = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))

        axs = np.atleast_1d(axs).flatten()

        for ax, img in zip(axs, imgs):
            ax.imshow(img)
            ax.axis('off')  
        
        for ax in axs[len(imgs):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()