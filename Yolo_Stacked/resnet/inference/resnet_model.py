import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms
from torchvision import models
from torch import nn

class Resnet_Classifier:
    def __init__(self, path):
        self.weights = path
        self.model = models.resnet34(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ]) 
        self.device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu" 
        pass

    def imshow(self, inp, title=None):
        """Display image for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated 

    def visualize_model_predictions(self, img_src):
        was_training = self.model.training
        self.model.eval()
        self.model.to(self.device)

        img = Image.open(img_src).convert("RGB")
        img = self.val_transform(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        with torch.no_grad():
            outputs = self.model(img)
            _, preds = torch.max(outputs, 1)

            ax = plt.subplot(2,2,1)
            ax.axis('off')
            ax.set_title(f'Predicted: {preds[0]}')
            self.imshow(img.cpu().data[0])

            self.model.train(mode=was_training)

    def predict(self, img_src):
        """
        Return:
          - pred_class (int)
          - confidence (float)
          - probs (np.array of shape [num_classes])
        """
        self.model.eval()
        img = self.val_transform(img_src)
        img = img.unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img)              # [1, 2]
            probs = torch.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, dim=1)

        return preds[0].item(), conf[0].item(), probs[0].cpu().numpy()

