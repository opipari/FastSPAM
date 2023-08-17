import os
import json
import argparse

from tqdm import tqdm
import math

from PIL import Image
import numpy as np
import torch

from panopticapi.utils import id2rgb

from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from MVPd.utils.MVPdHelpers import get_xy_depth, get_RT_inverse, get_pytorch3d_matrix, get_cameras, label_to_one_hot
from visualize import masks_to_panomasks


from segment_anything.modeling import Sam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

from video_panoptic_segmentation.metrics import utils as metric_utils


def get_dataset(dataset_config):
    return MVPDataset(root=dataset_config['root'],
                            split=dataset_config['split'],
                            training=dataset_config['training'],
                            window_size = dataset_config['window_size'])
    
def get_model(model_config, device):
    model_type = model_config['model_type']
    sam_checkpoint = model_config['sam_checkpoint']
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device=device)
    model = SamAutomaticMaskGenerator(sam, output_mode="coco_rle")
    return model


def evaluation_process(index, nprocs, config, output_dir):
    dataset = get_dataset(config['dataset'])
    nvideos_per_proc = math.ceil(len(dataset)/nprocs)
    subset_indices = torch.arange(index*nvideos_per_proc, min(((index+1)*nvideos_per_proc), len(dataset)))
    MVPdatasubset = torch.utils.data.Subset(dataset, subset_indices)

    model = get_model(config['model'], index)
    
    
    with torch.no_grad():
        for video in tqdm(MVPdatasubset):
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
                pred = model.generate(image)
                torch.save({"coco_rle":[pr['segmentation'] for pr in pred]}, os.path.join(out_dir, out_file))

            print("Finished processing", video_name)
            break
        

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    parser.add_argument("--output-path", dest="output_path")
    args = parser.parse_args()

    config = json.load(open(args.config_path, 'r'))

    num_gpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(evaluation_process, args=(num_gpus, config, args.output_path), nprocs=num_gpus)

    