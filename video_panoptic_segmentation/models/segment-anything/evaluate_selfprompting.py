import os
import json
import math

from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable

import numpy as np

import torch
import torch.nn as nn

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, video_collate
from MVPd.utils.MVPdHelpers import get_xy_depth, get_RT_inverse, get_pytorch3d_matrix, get_cameras, label_to_one_hot
from samselfprompting import SAMSelfPrompting

from video_panoptic_segmentation.metrics import utils as metric_utils


def get_dataset(dataset_config):
    return MVPDataset(root=os.path.join(dataset_config['root']),
                            split=dataset_config['split'],
                            training=dataset_config['training'],
                            window_size = dataset_config['window_size'])
    
def get_model(model_config, device):
    model_type = model_config['model_type']
    sam_checkpoint = os.path.join(model_config['sam_checkpoint'])
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device=device)
    
    model = SAMSelfPrompting(sam, 
                            prompts_per_object = 20,
                            objects_per_batch = 10
                           )

    return model

def evaluation_process(index, nprocs, config, output_dir):
    dataset = get_dataset(config['dataset'])
    nvideos_per_proc = math.ceil(len(dataset)/nprocs)
    subset_indices = torch.arange(index*nvideos_per_proc, min(((index+1)*nvideos_per_proc), len(dataset)))
    MVPdatasubset = torch.utils.data.Subset(dataset, subset_indices)

    model = get_model(config['model'], index)
    
    
    with torch.no_grad():
        for video in MVPdatasubset:
            model.reset_memory()
            
            for index, sample in enumerate(video):
                
                image = torch.tensor(sample['observation']['image']).permute(0,3,1,2).to('cuda')
                depth = torch.tensor(sample['observation']['depth']).unsqueeze(1).to('cuda')
                camera = get_cameras(sample['camera']['K'],
                                    sample['camera']['W2V_pose'],
                                    sample['meta']['image_size']).to('cuda')        

                out = model(image, depth, camera)
                torch.save(out, os.path.join(out_dir, out_file))

            print("Finished processing", video_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    parser.add_argument("--output-path", dest="output_path")
    args = parser.parse_args()

    config = json.load(open(args.config_path, 'r'))

    num_gpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(evaluation_process, args=(num_gpus, config, args.output_path), nprocs=num_gpus)

    