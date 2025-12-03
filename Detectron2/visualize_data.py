import os
import random
import cv2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.visualizer import Visualizer

"""
Visualize a few annotated samples to verify dataset.
"""

# paths relative to project root
DATA_ROOT = "data"
ANN_TRAIN = os.path.join(DATA_ROOT, "annotations", "instances_train.json")
ANN_VAL = os.path.join(DATA_ROOT, "annotations", "instances_val.json")
IMG_TRAIN = os.path.join(DATA_ROOT, "train", "imgs")
IMG_VAL = os.path.join(DATA_ROOT, "val", "imgs")
OUT_DIR = "samples"
N_SAMPLES = 3

os.makedirs(OUT_DIR, exist_ok=True)

def ensure_registered():
    # registers "train" and "val" datasets in Detectron2 if not already registered
    if "train" not in DatasetCatalog.list():
        register_coco_instances("train", {}, str(ANN_TRAIN), str(IMG_TRAIN))
    if "val" not in DatasetCatalog.list():
        register_coco_instances("val",   {}, str(ANN_VAL),   str(IMG_VAL))

def visualize_split(split_name, n_samples):
    # visualizes a few random images from the dataset split ('train' or 'val')
    dataset_dicts = DatasetCatalog.get(split_name)
    metadata = MetadataCatalog.get(split_name)

    # pick n random samples or all if fewer
    n = min(n_samples, len(dataset_dicts))
    picks = random.sample(dataset_dicts, n) if len(dataset_dicts) >= n else dataset_dicts

    for rec in picks:
        img_path = rec["file_name"]
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skip unreadable: {img_path}")
            continue

        vis = Visualizer(img[:, :, ::-1], metadata=metadata, scale=0.8)
        out = vis.draw_dataset_dict(rec)
        out_img = out.get_image()[:, :, ::-1]

        # save visualization
        base = os.path.splitext(os.path.basename(img_path))[0]
        out_name = f"{split_name}_{base}.jpg"
        out_path = os.path.join(OUT_DIR, out_name)
        cv2.imwrite(out_path, out_img)
        print("Saved:", out_path)

if __name__ == "__main__":
    ensure_registered()
    visualize_split("train", N_SAMPLES)
    visualize_split("val", N_SAMPLES)
    print(f"Done. Check the '{OUT_DIR}' folder.")