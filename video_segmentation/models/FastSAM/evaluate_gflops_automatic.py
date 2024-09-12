import json
import argparse

import torch
from fastsam import FastSAM, FastSAMPrompt

from fvcore.nn import FlopCountAnalysis


def get_model(model_config):
    model = FastSAM(model_config["checkpoint"])
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    args = parser.parse_args()

    config = json.load(open(args.config_path, 'r'))

    model = get_model(config['model'])
    
    im = torch.randn(1,3,config['model']['imgsz'],config['model']['imgsz'])
    gflops = FlopCountAnalysis(model.model, (im))
    print(f"GFlops: {gflops.total()/1e9}")
