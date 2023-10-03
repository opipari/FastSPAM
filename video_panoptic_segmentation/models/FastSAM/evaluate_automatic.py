import os
import json
import argparse

import time
from tqdm import tqdm
import math

from PIL import Image
import numpy as np
import torch

from panopticapi.utils import id2rgb

from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate

from fastsam import FastSAM, FastSAMPrompt

from video_panoptic_segmentation.metrics import utils as metric_utils


def get_dataset(dataset_config):
    dataset = MVPDataset(root=dataset_config["root"],
                        split=dataset_config["split"],
                        window_size = 0)

    return dataset
    
def get_model(model_config):
    model = FastSAM(model_config["checkpoint"])
    return model


def evaluation_process(index, nprocs, config, output_dir):

    dataset = get_dataset(config["dataset"])

    model = get_model(config['model'])


    if nprocs>0:
        is_per_proc = math.ceil(len(dataset)/nprocs)
        i_start = index*is_per_proc
        i_end = min((index+1)*is_per_proc, len(dataset))
        inds = list(range(i_start, i_end))
        dataset = torch.utils.data.Subset(dataset, inds)
        print(len(dataset))

    print("Within evaluation process")
    with torch.no_grad():
        for vi, video in enumerate(dataset):
            vitime = time.time()
            first_sample = next(iter(video))
            video_name = first_sample['meta']['video_name']
            out_dir = os.path.join(output_dir, config['experiment_name'], 'panomasksRLE', video_name)
            os.makedirs(out_dir, exist_ok=True)

            for index, sample in enumerate(video):
                # Load metadata
                video_name = sample['meta']['video_name']
                out_dir = os.path.join(output_dir, config['experiment_name'], 'panomasksRLE', video_name)
                out_file = sample['meta']['window_names'][0].split('.')[0]+'.pt'

                # Run SAM
                image = np.array(sample['observation']['image'][0]).astype(np.uint8) # 480 x 640 x 3

                input = Image.fromarray(image)
                input = input.convert("RGB")
                everything_results = model(
                    input,
                    device=0,
                    retina_masks=config["model"]["retina"],
                    imgsz=config["model"]["imgsz"],
                    conf=config["model"]["conf"],
                    iou=config["model"]["iou"]
                    )
                prompt_process = FastSAMPrompt(input, everything_results, device=0)
                ann = prompt_process.everything_prompt()

                coco_rle = [metric_utils.coco_encode_rle(rle) for rle in metric_utils.mask_to_rle_pytorch(ann.to(torch.bool))]
                torch.save({"coco_rle":coco_rle}, os.path.join(out_dir, out_file))


            print(f"Finished processing {vi}/{len(dataset)}: ", video_name, f"in {time.time()-vitime}s", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    parser.add_argument("--output-path", dest="output_path")
    args = parser.parse_args()

    config = json.load(open(args.config_path, 'r'))

    nprocs = 3
    torch.multiprocessing.spawn(evaluation_process, args=(nprocs, config, args.output_path), nprocs=nprocs)

    