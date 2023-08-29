import os
import json
import argparse

import importlib

import time
from tqdm import tqdm
import math

from PIL import Image
import numpy as np

import torch
import torchvision as tv
from torchvision import datapoints
import torchvision.transforms.v2.functional as F

import lightning as L

from MVPd2SA1B import MVPd2SA1B, SanitizeMasksPointsBoxes, RecomputeBoxes, ResamplePoints, RandomSamplePointsAndBoxes, RandomDropPointsOrBoxes

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from isam import ISam

from video_panoptic_segmentation.metrics import utils as metric_utils


def get_cfg(config_path):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    sys.modules["config"] = config
    spec.loader.exec_module(config)
    return config.InitConfig()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    parser.add_argument("--output-path", dest="output_path")
    args = parser.parse_args()

    cfg = get_cfg(args.config_path)

    num_devices = torch.cuda.device_count()
    print(f"Trying to use {num_devices} gpus")

    fabric = L.Fabric(
        devices=num_devices,
        loggers=L.fabric.loggers.TensorBoardLogger(os.path.join(cfg.output_dir, cfg.experiment_name), name='train_automatic_sam')
        )
    fabric.launch()


    sam = sam_model_registry[cfg.model_type](checkpoint=cfg.sam_checkpoint)
    isam = ISam(sam)
    optimizer = torch.optim.AdamW(sam.parameters(), lr=cfg.learning_rate, betas=cfg.adam_betas, weight_decay=cfg.weight_decay)
    isam, optimizer = fabric.setup(isam, optimizer)


    #
    # Dataset
    #
    pre_size = sam.image_encoder.img_size
    resize = (int(pre_size*(480/640)),pre_size)
    if cfg.use_augmentation:
        transform=tv.transforms.v2.Compose([
                            # tv.transforms.v2.RandomResizedCrop(size=(pre_size,pre_size), antialias=True),
                            tv.transforms.v2.Resize(size=resize),
                            tv.transforms.v2.ColorJitter(brightness=.5),
                            tv.transforms.v2.RandomGrayscale(p=0.1),
                            tv.transforms.v2.RandomHorizontalFlip(p=0.5),
                            SanitizeMasksPointsBoxes(min_size=(0.001*480*640)),
                            RecomputeBoxes(),
                            ResamplePoints(),
                            RandomSamplePointsAndBoxes(n_samples=cfg.num_mask_samples),
                            RandomDropPointsOrBoxes(p_points=0.5),
                            tv.transforms.v2.ToDtype(torch.float32),
                        ])
    else:
        transform=tv.transforms.v2.Compose([
                            # tv.transforms.v2.RandomResizedCrop(size=(pre_size,pre_size), antialias=True),
                            tv.transforms.v2.Resize(size=resize),
                            SanitizeMasksPointsBoxes(min_size=(0.001*480*640)),
                            RecomputeBoxes(),
                            ResamplePoints(),
                            RandomSamplePointsAndBoxes(n_samples=cfg.num_mask_samples),
                            RandomDropPointsOrBoxes(p_points=0.5),
                            tv.transforms.v2.ToDtype(torch.float32),
                        ])


    train_dataset = MVPd2SA1B(root = './video_panoptic_segmentation/datasets/MVPd/MVPd',
                        split = 'train',
                        transform=transform
                        )
    val_dataset = MVPd2SA1B(root = './video_panoptic_segmentation/datasets/MVPd/MVPd',
                        split = 'val',
                        transform=transform
                        )

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x)

    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)


    def get_lr(iter, learning_rate, warmup_iters, lr_decay_iters):
        if iter < warmup_iters:
            return learning_rate * iter / warmup_iters

        if iter>lr_decay_iters:
            return learning_rate * 0.1

        return learning_rate 


    def get_iteration_loss_train(isam, batch):
        isam.set_image(batch)
        
        iter_loss = 0
        for i in range(isam.interactive_iterations+1):
            batch, loss = isam.forward_interactive(batch, multimask_output=True)
            iter_loss += loss.detach().item()
            try:    
                fabric.backward(loss, retain_graph=True)
            except:
                print(f"[rank: {fabric.global_rank}] iterloss {i} {torch.cuda.memory_summary()}")
            del loss

        for i in range(isam.mask_iterations):
            batch, loss = isam.forward_interactive(batch, multimask_output=True)
            iter_loss += loss.detach().item()
            fabric.backward(loss, retain_graph=True)
            del loss
        
        iter_loss /= isam.interactive_iterations+isam.mask_iterations+1
        return iter_loss

    def get_iteration_loss_val(isam, batch):
        isam.set_image(batch)
        
        iter_loss = 0
        for i in range(isam.interactive_iterations+1):
            batch, loss = isam.forward_interactive(batch, multimask_output=True)
            iter_loss += loss.detach().item()

        for i in range(isam.mask_iterations):
            batch, loss = isam.forward_interactive(batch, multimask_output=True)
            iter_loss += loss.detach().item()
        
        iter_loss /= isam.interactive_iterations+isam.mask_iterations+1
        return iter_loss


    master_process = fabric.global_rank == 0
    if master_process:
        os.makedirs(os.path.join(cfg.output_dir, cfg.experiment_name), exist_ok=True)

    
    fabric.seed_everything(cfg.seed)
    iteration = 0
    best_val_loss = 10000
    while iteration < cfg.total_iterations:
        for batch_i, batch in enumerate(train_dataloader):

            lr = get_lr(iteration, learning_rate=cfg.learning_rate, warmup_iters=cfg.warmup_iters, lr_decay_iters=cfg.lr_decay_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            iter_loss = get_iteration_loss_train(isam, batch)
            optimizer.step()
            if iteration%10==0:
                fabric.log("loss", iter_loss, iteration)
                fabric.log("lr", lr, iteration)
                fabric.print(f"[rank: {fabric.global_rank}] Iter:{iteration}/{cfg.total_iterations} Loss:{iter_loss}")

            # if iteration%cfg.eval_every==0:
            #     with torch.no_grad():
            #         fabric.save(os.path.join(cfg.output_dir, cfg.experiment_name, f"checkpoints/checkpoint_{iteration}.ckpt"), isam.state_dict())

            #         avg_val_loss = 0
            #         eval_iteration = 0
            #         while eval_iteration < cfg.eval_iterations:
            #             for val_batch in val_dataloader:
            #                 iter_loss = get_iteration_loss_val(isam, val_batch)
            #                 avg_val_loss += iter_loss
            #                 if iter_loss<best_val_loss:
            #                     fabric.save(os.path.join(cfg.output_dir, cfg.experiment_name, f"checkpoints/best_model.ckpt"), isam.state_dict())
            #                     best_val_loss = iter_loss
            #                 if eval_iteration%10==0:
            #                     fabric.print(f"[rank: {fabric.global_rank}] Iter:{eval_iteration}/{cfg.eval_iterations} Val Loss:{iter_loss}")
            #                 eval_iteration+=1
            #                 if eval_iteration>=cfg.eval_iterations:
            #                     break
            #         avg_val_loss /= cfg.eval_iterations
            #         fabric.log("val_loss", avg_val_loss, iteration)

            iteration+=1
            if iteration>=cfg.total_iterations:
                break
    fabric.save(os.path.join(cfg.output_dir, cfg.experiment_name, f"checkpoints/final_checkpoint_{iteration}.ckpt"), isam.state_dict())

        