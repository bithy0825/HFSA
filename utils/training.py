import os
from loguru import logger
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    StepLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ReduceLROnPlateau,
)

from datasets import make_dataset
from models import make_model


def make_data_loader(spec, tag=""):
    if spec is None:
        return None

    dataset = make_dataset(spec["dataset"])
    dataset = make_dataset(spec["wrapper"], args={"dataset": dataset})

    logger.info(f"{tag} dataset: size = {len(dataset)}")
    """for k, v in dataset[0].items():
        logger.info(f"{tag} dataset: {k} = {v.shape if hasattr(v, 'shape') else v}")"""

    loader = DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=(tag == "train"),
        num_workers=12,
        pin_memory=True,
    )

    return loader


def make_data_loaders(cfg):
    train_loader = make_data_loader(cfg.get("train_dataset"), tag="train")
    val_loader = make_data_loader(cfg.get("val_dataset"), tag="val")
    return train_loader, val_loader


def make_optimizer(params, optimizer_spec, load_sd=False):
    Optimizer = {
        "sgd": SGD,
        "adam": Adam,
        "adamw": AdamW,
    }[optimizer_spec["name"]]
    optimizer = Optimizer(
        params,
        **optimizer_spec["args"],
    )
    if load_sd:
        optimizer.load_state_dict(optimizer_spec["sd"])

    return optimizer


def make_scheduler(optimizer, scheduler_spec, load_sd=False):
    Scheduler = {
        "step": StepLR,
        "multistep": MultiStepLR,
        "cosine": CosineAnnealingLR,
        "cosine_warm": CosineAnnealingWarmRestarts,
        "reduce_on_plateau": ReduceLROnPlateau,
    }[scheduler_spec["name"]]
    scheduler = Scheduler(
        optimizer,
        **scheduler_spec["args"],
    )
    if load_sd:
        scheduler.load_state_dict(scheduler_spec["sd"])

    return scheduler


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return f"{tot / 1e6:.2f}M"
        else:
            return f"{tot / 1e3:.2f}K"
    return tot


def prepare_training(cfg, device):
    if os.path.exists(cfg.get("resume")):
        sv_file = torch.load(cfg["resume"], map_location=device, weights_only=True)
        model_spec = sv_file["model"].copy()
        state_dict = model_spec["sd"]

        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith("_orig_mod."):
                    new_key = key[10:]
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            model_spec["sd"] = new_state_dict

        model = make_model(model_spec, load_sd=True).to(device)
        optimizer = make_optimizer(
            model.parameters(), sv_file["optimizer"], load_sd=True
        )
        start_epoch = sv_file["epoch"] + 1
        if sv_file.get("scheduler") is not None:
            scheduler = make_scheduler(optimizer, sv_file["scheduler"], load_sd=True)
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                for _ in range(start_epoch - 1):
                    scheduler.step()
        else:
            scheduler = None

        if sv_file.get("max_val") is not None:
            max_val = sv_file["max_val"]

    else:
        model = make_model(cfg["model"]).to(device)
        optimizer = make_optimizer(model.parameters(), cfg["optimizer"])
        scheduler = (
            make_scheduler(optimizer, cfg["scheduler"])
            if cfg.get("scheduler")
            else None
        )
        start_epoch = 1
        max_val = 1e-18

    logger.info(
        f"model: {model.__class__.__name__}, params = {compute_num_params(model, text=True)}"
    )

    return model, optimizer, scheduler, start_epoch, max_val
