import os
import cv2
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
import random
import glob
from datetime import datetime

# settings

# trained model weights
WEIGHTS = "./output/model_final.pth"

# text file with class names
CLASS_NAMES_FILE = "class.names"

# image to test
# IMAGE_PATH = "./data/val/imgs/check_001_000_G1_2.jpg"
IMAGE_PATH = random.choice(glob.glob("data/val/imgs/*.jpg"))
IMAGE_DIR = "data/val/imgs"

# score threshold for display
THRESH = 0.40
NUM_SAMPLES = 10

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
all_img = glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
random_imgs = random.sample(all_img, min(NUM_SAMPLES, len(all_img)))

# output folder
os.makedirs("output_saved", exist_ok=True)
run_dir = "output_saved"

for idx, img_path in enumerate(random_imgs, start=1):
    img = cv2.imread(img_path)
    assert img is not None, f"Image not found: {img_path}"

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
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    out_path = os.path.join(run_dir, f"{base_name}_prediction.jpg")
    cv2.imwrite(out_path, img)
    print(f"Saved:", os.path.abspath(out_path))

print(f"Done saved: {len(random_imgs)} predictions in output_saved/")
