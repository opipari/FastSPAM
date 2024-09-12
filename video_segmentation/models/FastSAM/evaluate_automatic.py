import os
import json
import argparse

import time
from tqdm import tqdm
import math

from PIL import Image
import numpy as np
import torch

#from fvcore.nn import FlopCountAnalysis

from panopticapi.utils import id2rgb

from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate

from fastsam import FastSAM, FastSAMPrompt

from video_segmentation.metrics import utils as metric_utils


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
    
    rand_inds = [1367,  100,  983,   83, 1586,  896,  683, 1598, 1092, 1020,  435,  747,
        1497,  484,  473,  367,  622,   14, 1290, 1414,  459,  740,  111, 1383,
        1189, 1698, 1280, 1584,  286,  229, 1279,  463, 1396, 1305,  500,  214,
        1513,  108,  536, 1729,  240, 1184, 1061,  722,  513,  650,   59,  128,
         180, 1028,   96,  970, 1488, 1470, 1128,  615,  274, 1155,  519,  280,
         938, 1547,  234,  971,  620,  677,  659,  227, 1135, 1001,  150,  586,
         825, 1541,    4,  178, 1285,  209,  966,  153,  857,  580,  633,  954,
        1520,  852, 1580,  348, 1457,  223,  205,  824,   56,  511, 1400,  879,
        1632,  116, 1007, 1145,  717, 1207, 1164,  772, 1002, 1315, 1114, 1206,
        1011,   44, 1374, 1505, 1269, 1049,  351,  373, 1510,  177, 1427, 1193,
         853,   41, 1583,  648, 1391,  829, 1485, 1682, 1362,   64, 1328, 1262,
        1228, 1499, 1492,  778, 1641, 1297,  791, 1349,  483, 1168, 1324, 1201,
         979,   88, 1010, 1590, 1208, 1132, 1714,  173,  893, 1460,  477, 1096,
         847,  216, 1601,  624,  119,  667,  527,  237,  221, 1496,  322,  989,
        1614,  329,  305,  122, 1401,  251]
    # print("Within evaluation process")
    with torch.no_grad():
        for vi, video in enumerate(tqdm(dataset, position=0, disable=index!=0)):
            if vi not in rand_inds:
                continue
            # vitime = time.time()
            first_sample = next(iter(video))
            video_name = first_sample['meta']['video_name']
            out_dir = os.path.join(output_dir, config['experiment_name'], 'panomasksRLE', video_name)
            os.makedirs(out_dir, exist_ok=True)



            for sample in tqdm(video, position=1, disable=index!=0):
                # Load metadata
                video_name = sample['meta']['video_name']
                out_dir = os.path.join(output_dir, config['experiment_name'], 'panomasksRLE', video_name)
                out_file = sample['meta']['window_names'][0].split('.')[0]+'.pt'

                if os.path.exists(os.path.join(out_dir, out_file)):
                    continue

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
                if len(ann)>0:
                    ann = ann.to(torch.bool)
                    ann = metric_utils.mask_to_rle_pytorch(ann)
                coco_rle = [metric_utils.coco_encode_rle(rle) for rle in ann]
                torch.save({"coco_rle":coco_rle}, os.path.join(out_dir, out_file))


            # print(f"Finished processing {vi}/{len(dataset)}: ", video_name, f"in {time.time()-vitime}s", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    parser.add_argument("--output-path", dest="output_path")
    args = parser.parse_args()

    config = json.load(open(args.config_path, 'r'))

    nprocs = 3
    # torch.multiprocessing.spawn(evaluation_process, args=(nprocs, config, args.output_path), nprocs=nprocs)

    evaluation_process(0,0,config,args.output_path)
