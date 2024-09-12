import os
import json
from copy import deepcopy

import torch
import torch.nn as nn

from fvcore.nn import FlopCountAnalysis

from segment_anything.modeling import Sam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def get_model(model_config, device):
    model_type = model_config['model_type']
    sam_checkpoint = model_config['sam_checkpoint']
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device=device)
    model = SamAutomaticMaskGenerator(sam, output_mode="coco_rle", points_per_side=model_config['points_per_side'])
    return model


def get_flop(config):
    
    sam_model = get_model(config['model'], 'cpu')


    print("Within evaluation process")
    with torch.no_grad():
        num_prompts = 32*32
        inputs = [{
                'image': torch.randn(3, 768, 1024).to(sam_model.predictor.model.device),
                'original_size': (480, 640),
                'point_coords': torch.randn(1,num_prompts,2).to(sam_model.predictor.model.device),
                'point_labels': torch.randn(1,num_prompts).to(sam_model.predictor.model.device),
                'boxes': None,
                'mask_inputs': None
            }]
        multimask = True
        
        gflops = FlopCountAnalysis(sam_model.predictor.model, (inputs, multimask))
        gflops = gflops.total() / 1e9

    return gflops

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    args = parser.parse_args()

    config = json.load(open(args.config_path, 'r'))

    gflops = get_flop(config)
    print(f"GFlops: {gflops}")
    
