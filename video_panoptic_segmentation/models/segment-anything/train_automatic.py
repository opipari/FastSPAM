import os
import json
import argparse

import time
from tqdm import tqdm
import math

from PIL import Image
import numpy as np

import torch
import lightning as L

from MVPd2SA1B import MVPd2SA1B

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from video_panoptic_segmentation.metrics import utils as metric_utils



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    parser.add_argument("--output-path", dest="output_path")
    args = parser.parse_args()

    # config = json.load(open(args.config_path, 'r'))

    num_gpus = torch.cuda.device_count()

    fabric = L.Fabric()
    fabric.launch()

    
    dataset = MVPd2SA1B(root = './video_panoptic_segmentation/datasets/MVPd/MVPd',
                        split = 'train')

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    dataloader = fabric.setup_dataloaders(dataloader)


    for batch_i, batch in enumerate(dataloader):
        print(batch['image'].shape, batch['points'].shape)
        print(batch['image'].device)
        if batch_i>10:
            break