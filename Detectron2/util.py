import os
import warnings
import torch
import detectron2
from detectron2.engine import DefaultTrainer, hooks
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2 import model_zoo
from loss import ValidationLoss

# ignore known torch meshgrid warning
warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")

# print environment versions for verification
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1] if "+" in torch.__version__ else "cpu"

print("torch:", TORCH_VERSION, "; cuda:", CUDA_VERSION, "| CUDA available:", torch.cuda.is_available())
print("detectron2:", detectron2.__version__)


# ---
# This section is testing the example
# ---
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file(
#     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# ))
# cfg.MODEL.DEVICE = "cpu"
# cfg.MODEL.WEIGHTS = ""
#
# model = build_model(cfg).eval()
# n_params = sum(p.numel() for p in model.parameters())
# print("✅ Built model on", cfg.MODEL.DEVICE, "with params:", n_params)
#
# IMG_PATH = "/Users/thyatran/Personal/Automation Anywhere Proj/input.jpg"  # move image OUT of .venv
# WEIGHTS  = "/Users/thyatran/Personal/Automation Anywhere Proj/model_final_f10217.pkl"
#
# assert os.path.exists(IMG_PATH), f"Image not found: {IMG_PATH}"
# assert os.path.isfile(WEIGHTS),  f"Weights not found: {WEIGHTS} (download locally first)"
# im = cv2.imread(IMG_PATH); assert im is not None, f"cv2 failed to read: {IMG_PATH}"
#
# cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file(
#     "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
# ))
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
# cfg.MODEL.DEVICE = "cpu"
# cfg.MODEL.WEIGHTS = WEIGHTS
#
# predictor = DefaultPredictor(cfg)
# outputs = predictor(im)
# inst = outputs["instances"].to("cpu")
#
# print("✅ Inference OK")
# print("num detections:", len(inst))
# print("classes:", inst.pred_classes.tolist() if len(inst) else [])
#
# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]) if len(cfg.DATASETS.TRAIN)>0 else None)
# out = v.draw_instance_predictions(inst)
# cv2.imwrite("output_vis.jpg", out.get_image()[:, :, ::-1])
# print("Saved: output_vis.jpg")

# function to register datasets in Detectron2
def register_datasets(root_dir, class_list_file):
    """
    Registers the train and validation datasets and returns the number of detected classes
    """
    # paths to json annotation files and image directories
    ann_train = os.path.join(root_dir, "annotations", "instances_train.json")
    ann_val = os.path.join(root_dir, "annotations", "instances_val.json")
    img_train = os.path.join(root_dir, "train", "imgs")
    img_val = os.path.join(root_dir, "val", "imgs")

    # verify all files and folders exist
    assert os.path.isfile(ann_train), f"Missing: {ann_train}"
    assert os.path.isfile(ann_val), f"Missing: {ann_val}"
    assert os.path.isdir(img_train), f"Missing: {img_train}"
    assert os.path.isdir(img_val), f"Missing: {img_val}"

    # read class names from text file ("signature", "amount", "amount_words", etc.)
    with open(class_list_file, "r") as f:
        classes_ = [line.strip() for line in f if line.strip()]

    # register datasets in Detectron2's DatasetCatalog
    register_coco_instances("train", {}, ann_train, img_train)
    register_coco_instances("val", {}, ann_val, img_val)

    # attach readable class names for visualization
    MetadataCatalog.get("train").set(thing_classes=classes_)
    MetadataCatalog.get("val").set(thing_classes=classes_)

    print(f"Registered datasets. #classes = {len(classes_)} | classes = {classes_}")
    return len(classes_)


# function build a Detectron2 configuration
def build_cfg(output_dir, learning_rate, batch_size, iterations, checkpoint_period, model, device, nmr_classes):
    """
    Builds a Detectron2 training configuration.

    Arguments:
        output_dir (str): folder to save checkpoints
        learning_rate (float): base learning rate
        batch_size (int): number of images per batch
        iterations (int): total training iterations
        checkpoint_period (int): save checkpoint every n iterations
        model (str): model architecture name from model_zoo
        device (str): "cpu" or "cuda"
        nmr_classes (int): number of output classes
    Returns:
        cfg: config object ready for training
    """

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(model))

    # solver (optimizer) settings
    cfg.SOLVER.BASE_LR = learning_rate  # Base learning rate

    # max number of training iterations
    cfg.SOLVER.MAX_ITER = iterations

    # batch size used by the solver, number of images per batch
    cfg.SOLVER.IMS_PER_BATCH = batch_size

    # the checkpoint period (save frequency)
    cfg.SOLVER.CHECKPOINT_PERIOD = checkpoint_period

    # print logs every iteration
    cfg.SOLVER.LOG_PERIOD = 1

    # evaluate frequently
    cfg.TEST.EVAL_PERIOD = 1

    # the learning rate scheduler steps to any empty list, which means the learning rate will not be decayed
    cfg.SOLVER.STEPS = []

    # set the training and validation datasets and exclude the test
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ()

    # model device and weights
    cfg.MODEL.DEVICE = "cpu" if device.lower() == "cpu" else "cuda"
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)

    # ROI (region of interest) head configuration
    cfg.MODEL.RETINANET.NUM_CLASSES = nmr_classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = nmr_classes

    # output directory for logs and checkpoint
    cfg.OUTPUT_DIR = output_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    return cfg


def train(output_dir, data_dir, class_list_file, learning_rate, batch_size, iterations, checkpoint_period, device,
          model):
    """
    Train a Detectron2 model on a custom dataset.
    """
    # register dataset and get number of classes
    nmr_classes = register_datasets(data_dir, class_list_file)

    # build training configuration
    cfg = build_cfg(output_dir=output_dir,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    iterations=iterations,
                    checkpoint_period=checkpoint_period,
                    model=model,
                    device=device,
                    nmr_classes=nmr_classes, )

    # initialize trainer
    trainer = DefaultTrainer(cfg)

    # attach validation loss monitoring
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])

    # fix hook order so validation loss prints correctly
    trainer._hooks = [h for h in trainer._hooks if not isinstance(h, hooks.PeriodicWriter)]
    trainer.register_hooks([hooks.PeriodicWriter(trainer.build_writers(), period=1)])

    # resume training from a checkpoint (if exists) or start fresh
    trainer.resume_or_load(resume=True)

    # train
    print("Starting training...")
    trainer.train()
