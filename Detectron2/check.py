import os
import json
import glob


def must_exist(p):
    """
    Ensures that a given file or directory exists
    If not, it raises error and stops the program
    """
    assert os.path.exists(p), f"Missing: {p}"


# check that all required project files and directories are present
must_exist("./class.names")
must_exist("data/annotations/instances_train.json")
must_exist("data/annotations/instances_val.json")
must_exist("data/train/imgs")
must_exist("data/val/imgs")

# count how many images are in the training and validation folders
n_train = len(glob.glob("data/train/imgs/*.jpg"))
n_val = len(glob.glob("data/val/imgs/*.jpg"))

# number of images found in each set
print("Train images:", n_train, "| Val images:", n_val)

# read and print all category names from the training json annotation file
with open("data/annotations/instances_train.json") as f:
    data = json.load(f)
    cats = sorted(data["categories"], key=lambda c: c["id"])

# category names
print("Class names:", [c["name"] for c in cats])
