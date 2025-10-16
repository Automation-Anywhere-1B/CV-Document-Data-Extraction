import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import random
import glob

# settings

# trained model weights
WEIGHTS = "./output/model_final.pth"

# text file with class names
CLASS_NAMES_FILE = "class.names"

# image to test
# IMAGE_PATH = "./data/val/imgs/check_001_000_G1_2.jpg"
IMAGE_PATH = random.choice(glob.glob("data/val/imgs/*.jpg"))

# score threshold for display
THRESH = 0.40

# load class names from file
with open(CLASS_NAMES_FILE, "r") as f:
    CLASS_NAMES = [ln.strip() for ln in f if ln.strip()]
NUM_CLASSES = len(CLASS_NAMES)

# buid Detectron2 configuration, match training setup
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = WEIGHTS
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.0
cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5

# create predictor for inference
predictor = DefaultPredictor(cfg)

# load input image
img = cv2.imread(IMAGE_PATH)
assert img is not None, f"Image not found: {IMAGE_PATH}"

# run inference
outputs = predictor(img)
inst = outputs["instances"].to("cpu")
boxes = inst.pred_boxes.tensor.numpy().tolist()     # predicted bounding boxes
scores = inst.scores.numpy().tolist()               # confidence score
classes = inst.pred_classes.numpy().tolist()        # class indices

# draw predictions on the image
for (box, score, cls) in zip(boxes, scores, classes):
    if score < THRESH:
        continue
    x1, y1, x2, y2 = map(int, box)
    label = CLASS_NAMES[cls] if 0 <= cls < NUM_CLASSES else f"id{cls}"

    # draw bounding box and label
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    text = f"{label} {int(score*100)}%"
    # text bg
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 0, 255), -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

# save the result and display it
out_path = "output_saved/prediction.jpg"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
cv2.imwrite(out_path, img)
print(f"Saved: {out_path}")

# show prediction window
cv2.imshow("Prediction", img)
while True:
    k = cv2.waitKey(10) & 0xFF
    if k in (27, ord('q')):
        break
cv2.destroyAllWindows()
