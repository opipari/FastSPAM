import json
import argparse

from fastsam import FastSAM, FastSAMPrompt
from ultralytics.yolo.utils.torch_utils import get_flops


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
    gflops = get_flops(model.model, imgsz=config["model"]["imgsz"])

    print(f"GFlops: {gflops}")
