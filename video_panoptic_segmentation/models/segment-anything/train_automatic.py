import os
import json
import argparse

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

from MVPd2SA1B import MVPd2SA1B, SanitizeMasksPointsBoxes, RecomputeBoxes, ResamplePoints, RandomDropPointsOrBoxes

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from isam import ISam

from video_panoptic_segmentation.metrics import utils as metric_utils



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    parser.add_argument("--output-path", dest="output_path")
    args = parser.parse_args()

    # config = json.load(open(args.config_path, 'r'))

    num_gpus = torch.cuda.device_count()

    fabric = L.Fabric(loggers=L.fabric.loggers.TensorBoardLogger(args.output_path, name='train_automatic_sam'))
    fabric.launch()


    model_type = "vit_b"
    sam_checkpoint = "./video_panoptic_segmentation/models/segment-anything/segment-anything/checkpoints/sam_vit_b_01ec64.pth"
    
    use_augmentation = False
    total_iterations = 5000
    eval_every = 1000
    learning_rate = 8e-4
    adam_betas = (0.9, 0.999)
    weight_decay = 0.1
    warmup_iters = 250
    lr_decay_iters = 3000
    seed = 0


    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    isam = ISam(sam)
    optimizer = torch.optim.AdamW(sam.parameters(), lr=learning_rate, betas=adam_betas, weight_decay=weight_decay)
    isam, optimizer = fabric.setup(isam, optimizer)


    #
    # Dataset
    #
    pre_size = sam.image_encoder.img_size
    resize = (int(pre_size*(480/640)),pre_size)
    if use_augmentation:
        transform=tv.transforms.v2.Compose([
                            # tv.transforms.v2.RandomResizedCrop(size=(pre_size,pre_size), antialias=True),
                            tv.transforms.v2.Resize(size=resize),
                            tv.transforms.v2.ColorJitter(brightness=.5),
                            tv.transforms.v2.RandomGrayscale(p=0.1),
                            tv.transforms.v2.RandomHorizontalFlip(p=0.5),
                            SanitizeMasksPointsBoxes(min_size=(0.001*480*640)),
                            RecomputeBoxes(),
                            ResamplePoints(),
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
                            RandomDropPointsOrBoxes(p_points=0.5),
                            tv.transforms.v2.ToDtype(torch.float32),
                        ])


    dataset = MVPd2SA1B(root = './video_panoptic_segmentation/datasets/MVPd/MVPd',
                        split = 'train',
                        transform=transform
                        )

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, collate_fn=lambda x: x)
    dataloader = fabric.setup_dataloaders(dataloader)


    

    def get_lr(iter, learning_rate = learning_rate, warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters):
        if iter < warmup_iters:
            return learning_rate * iter / warmup_iters

        if iter>lr_decay_iters:
            return learning_rate * 0.1

        return learning_rate 


    master_process = fabric.global_rank == 0

    if master_process:
        os.makedirs(args.output_path, exist_ok=True)

    fabric.seed_everything(seed)

    iteration = 0
    while iteration < total_iterations:
        for batch_i, batch in enumerate(dataloader):

            lr = get_lr(iteration)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            
            isam.set_image(batch)
            
            iter_loss = 0
            for i in range(isam.interactive_iterations+1):
                batch, loss = isam.forward_interactive(batch, multimask_output=True)
                iter_loss += loss.detach().item()
                fabric.backward(loss, retain_graph=True)
                del loss

            for i in range(isam.mask_iterations):
                batch, loss = isam.forward_interactive(batch, multimask_output=True)
                iter_loss += loss.detach().item()
                fabric.backward(loss, retain_graph=True)
                del loss
            
            iter_loss /= isam.interactive_iterations+isam.mask_iterations+1
            fabric.log("loss", iter_loss)
            fabric.print(f"{iteration}/{total_iterations} Loss:{iter_loss}")
            optimizer.step()

            if iteration%eval_every==0:
                state = {
                    "model": isam,
                    "optimizer": optimizer,
                    "iteration": iteration,
                }

                # Instead of `torch.save(...)`
                fabric.save(os.path.join(args.output_path, "checkpoints/checkpoint_{iteration}.ckpt"), state)

            iteration+=1
            if iteration>=total_iterations:
                break
            
        
        # import matplotlib
        # import matplotlib.pyplot as plt

        # print(batch[0]['image'].shape, batch[0]['image'].dtype, batch[0]['image'].min(), batch[0]['image'].max())
        # plt.imshow(batch[0]['image'].permute(1,2,0).cpu()/255)
        # plt.show()

        # fig, ax = plt.subplots()
        # i=0

        # ax.imshow(batch[0]['masks'][i].cpu())
        
        # if 'boxes' in batch[0]:
        #     rect = matplotlib.patches.Rectangle((batch[0]['boxes'].cpu()[i][0], batch[0]['boxes'].cpu()[i][1]), batch[0]['boxes'].cpu()[i][2]-batch[0]['boxes'].cpu()[i][0], batch[0]['boxes'].cpu()[i][3]-batch[0]['boxes'].cpu()[i][1], linewidth=1, edgecolor='r', facecolor='none')
        #     ax.add_patch(rect)
        # if 'point_coords' in batch[0]:
        #     print(batch[0]['point_coords'].shape)
        #     circle = plt.Circle((batch[0]['point_coords'].cpu()[i,0][0], batch[0]['point_coords'].cpu()[i,0][1]), 10, facecolor='r')
        #     ax.add_artist(circle)

        # plt.show()

