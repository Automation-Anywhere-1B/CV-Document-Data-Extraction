from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import io
import itertools
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
        
        images = []
        for img in imgs:
            # Case 1: already a PIL image
            if isinstance(img, Image.Image):
                images.append(img)

            # Case 2: numpy array 
            elif isinstance(img, np.ndarray):
                images.append(img)

            # Case 3: raw bytes
            elif isinstance(img, (bytes, bytearray)):
                pil_img = Image.open(io.BytesIO(img)).convert("RGB")
                images.append(pil_img)

        pred = self.model(images)
        self.predictions.append(pred)

        # --- Debug here ---
        # print(f"Got {len(pred)} result objects.")
        # r = pred[0]
        # print("Detected class IDs:", r.boxes.cls.cpu().numpy())
        # print("Class mapping:", r.names)
        # ------------------

        return pred
    
    #returned cropped signature image of the last prediction
    def signatures_last_prediction(self):
        if not self.predictions:
            return
        batch = self.predictions[-1]
        r = batch[0]
        img = r.orig_img
        boxes = r.boxes
        names = r.names  
        filtered = boxes[boxes.cls == 0]

        # Collect all class IDs that are signature-like
        sig_ids = [i for i, name in names.items() if name.startswith("signature")]

        if not sig_ids:
            return []

        cls_np = boxes.cls.cpu().numpy()
        mask = np.isin(cls_np, sig_ids)
        filtered = boxes[mask]

        if len(filtered) == 0:
            return []

        crops = []
        for b in filtered:
            x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().astype(int)
            crop = img[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop_rgb)
            crops.append(crop)
        return crops

    #returns np image array
    def visualize_last_prediction(self):
        if not self.predictions:
            return
        batch = self.predictions[-1]    
        imgs = [batch[i].plot()[::, ::, ::-1] for i in range(len(batch))]
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
            imgs.extend([batch[i].plot()[::, ::, ::-1] for i in range(len(batch))])
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