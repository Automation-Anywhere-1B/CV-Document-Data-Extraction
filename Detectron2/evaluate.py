import os
import json
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_test_loader
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset

DATA_DIR = "./data"
VAL_NAME = "val"
VAL_JSON = f"{DATA_DIR}/annotations/instances_val.json"
VAL_IMG_DIR = f"{DATA_DIR}/val/imgs"
WEIGHTS = "./output/model_final.pth"
CLASS_NAMES_FILE = "./class.names"
CONFIG_NAME = "COCO-Detection/retinanet_R_101_FPN_3x.yaml"

DEVICE = "cpu"

def main():

    assert os.path.exists(WEIGHTS), "Missing trained weights"
    assert os.path.exists(VAL_JSON), "Missing val annotations"
    assert os.path.exists(VAL_IMG_DIR), "Missing val images"
    assert os.path.exists(CLASS_NAMES_FILE), "Missing class.names"

    # register dataset
    if VAL_NAME not in DatasetCatalog.list():
        register_coco_instances(VAL_NAME, {}, VAL_JSON, VAL_IMG_DIR)

    # build config
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(CONFIG_NAME))
    cfg.MODEL.WEIGHTS = WEIGHTS
    cfg.MODEL.DEVICE = DEVICE

    with open(CLASS_NAMES_FILE) as f:
        classes = [c.strip() for c in f if c.strip()]
    cfg.MODEL.RETINANET.NUM_CLASSES = len(classes)

    cfg.DATASETS.TEST = (VAL_NAME,)
    cfg.DATALOADER.NUM_WORKERS = 0

    cfg.INPUT.MIN_SIZE_TEST = 640
    cfg.INPUT.MAX_SIZE_TEST = 960
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    # load model
    model = build_model(cfg)
    DetectionCheckpointer(model).load(WEIGHTS)
    model.eval()

    evaluator = COCOEvaluator(VAL_NAME, output_dir="./output")
    loader = build_detection_test_loader(cfg, VAL_NAME)

    print("\nRunning evaluation on validation set...")
    results = inference_on_dataset(model, loader, evaluator)

    print(json.dumps(results, indent=2))
    with open("./output/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved eval_results.json")

if __name__ == "__main__":
    main()
