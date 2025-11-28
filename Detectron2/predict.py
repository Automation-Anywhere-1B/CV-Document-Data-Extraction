import cv2
from pathlib import Path

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

THIS_DIR = Path(__file__).resolve().parent

# trained model weights: Detectron2/output/model_final.pth
WEIGHTS = THIS_DIR / "output" / "model_final.pth"

# class names file: Detectron2/class.names
CLASS_NAMES_FILE = THIS_DIR / "class.names"

THRESH = 0.40

# predictor for streamlit
with open(CLASS_NAMES_FILE, "r") as f:
    CLASS_NAMES = [ln.strip() for ln in f if ln.strip()]
NUM_CLASSES = len(CLASS_NAMES)

cfg = get_cfg()
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml")
)
cfg.MODEL.WEIGHTS = str(WEIGHTS)
cfg.MODEL.DEVICE = "cpu"
cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.0
cfg.MODEL.RETINANET.NMS_THRESH_TEST = 0.5

predictor = DefaultPredictor(cfg)

def get_trained_predictor():
    """
    Expose the trained predictor + class names
    so Streamlit can reuse them.
    """
    return predictor, CLASS_NAMES
