from detectron2.engine import HookBase
from detectron2.data import build_detection_train_loader
import detectron2.utils.comm as comm
import torch

# makes the trainer run a quick forward pass on the validation set after each training step,
# measures how well the model is doing (without updating weights),
# and logs the validation loss alongside the training loss to track both during training.

class ValidationLoss(HookBase):
    """
    Compute validation loss on the 'val' dataset after each training step.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()

        # problems
        # Use the validation dataset for checking model performance during training.
        # If a TEST dataset is already defined in the config, use that instead.
        # Otherwise, default to the "val" dataset.
        if len(cfg.DATASETS.TEST) > 0:
            self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        else:
            self.cfg.DATASETS.TRAIN = ("val",)

        self._make_loader()

    def _make_loader(self):
        self._loader = iter(build_detection_train_loader(self.cfg))

    def after_step(self):
        """Run one validation forward pass and log losses."""
        try:
            data = next(self._loader)
        except StopIteration:
            self._make_loader()
            data = next(self._loader)

        with torch.no_grad():
            loss_dict = self.trainer.model(data)

        # ensure losses are finite
        total = sum(loss_dict.values())
        assert torch.isfinite(total).all(), f"Non-finite val loss: {loss_dict}"

        # reduce across workers
        reduced = {"val_" + k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
        total_val = sum(reduced.values())

        # only main process writes to storage
        if comm.is_main_process():
            storage = self.trainer.storage
            storage.put_scalar("total_val_loss", total_val)
            for k, v in reduced.items():
                storage.put_scalar(k, v)
