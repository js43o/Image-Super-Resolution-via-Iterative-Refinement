import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from core.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="config/sr_sr3_16_128.json",
        help="JSON file for configuration",
    )
    parser.add_argument(
        "-p",
        "--phase",
        type=str,
        choices=["train", "val"],
        help="Run either train(training) or val(generation)",
        default="train",
    )
    parser.add_argument("-gpu", "--gpu_ids", type=str, default=None)
    parser.add_argument("-debug", "-d", action="store_true")
    parser.add_argument("-enable_wandb", action="store_true")
    parser.add_argument("-log_wandb_ckpt", action="store_true")
    parser.add_argument("-log_eval", action="store_true")

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(
        None, opt["path"]["log"], "train", level=logging.INFO, screen=True
    )
    Logger.setup_logger("val", opt["path"]["log"], "val", level=logging.INFO)
    logger = logging.getLogger("base")
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt["path"]["tb_logger"])

    # Initialize WandbLogger
    if opt["enable_wandb"]:
        import wandb

        wandb_logger = WandbLogger(opt)
        wandb.define_metric("validation/val_step")
        wandb.define_metric("epoch")
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train" and args.phase != "val":
            train_set = Data.create_dataset(dataset_opt, phase)
            train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
        elif phase == "val":
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(val_set, dataset_opt, phase)
    logger.info("Initial Dataset Finished")

    # model
    diffusion = Model.create_model(opt)
    logger.info("Initial Model Finished")

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt["train"]["n_iter"]

    if opt["path"]["resume_state"]:
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                current_epoch, current_step
            )
        )

    diffusion.set_new_noise_schedule(
        opt["model"]["beta_schedule"][opt["phase"]], schedule_phase=opt["phase"]
    )
