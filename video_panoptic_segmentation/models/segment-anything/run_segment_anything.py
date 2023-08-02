import os
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from scipy.optimize import linear_sum_assignment

import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

#from zeroshot_rgbd.datasets.VaryingPerspectiveDataset import VaryingPerspectiveDataset
from zeroshot_rgbd.datasets.ActiveIlluminationDataset import SceneDataset
#from habitat_sim.utils.common import d3_40_colors_rgb



def save_anns(prefix, dest_dir, img, anns):
    fig = plt.figure(figsize=(20,20))
    plt.axis('off')
    plt.imshow(img)

    if len(anns) > 0:
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:,:,3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
        ax.imshow(img)
    #plt.savefig(os.path.join(dest_dir, str(prefix)+".SAM.jpg"))
    fig.canvas.draw()
    res = Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
    res.save(os.path.join(dest_dir, str(prefix)+".SAM.png"))
    plt.close()



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", dest="img_dir")
    parser.add_argument("--dest-dir", dest="dest_dir")
    parser.add_argument("--sam-checkpoint", dest="sam_checkpoint")
    parser.add_argument("--model-type", dest="model_type")
    parser.set_defaults(img_dir="./zero_shot_scene_segmentation/datasets/renders/example",
                    dest_dir="./zero_shot_scene_segmentation/models/segment-anything/output/",
                    sam_checkpoint="./zero_shot_scene_segmentation/models/segment-anything/segment-anything/checkpoints/sam_vit_h_4b8939.pth",
                    model_type="vit_h")
    args, _ = parser.parse_known_args()
    img_dir = args.img_dir
    dest_dir = args.dest_dir
    sam_checkpoint = args.sam_checkpoint
    model_type = args.model_type

    device = "cuda"


    if not os.path.isdir(dest_dir):
        os.makedirs(dest_dir)

    
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    illumination=0
    dataset = SceneDataset(root=img_dir, illumination=illumination)

    

    for idx in range(len(dataset)):
        if idx%10==0:
            print("processing",idx,"/",len(dataset))
        image, label, scene_name, view_prefix = dataset[idx]
        image = image.permute(1,2,0).numpy()
        
        os.makedirs(os.path.join(dest_dir, scene_name), exist_ok=True)

        masks = mask_generator.generate(image)
        save_anns(view_prefix+'.RGB.'+f"{illumination:010}", os.path.join(dest_dir, scene_name), image, masks)
        masks = torch.stack([torch.tensor(ann['segmentation']) for ann in masks])
        torch.save(masks, os.path.join(dest_dir, scene_name,view_prefix+'.RGB.'+f"{illumination:010}.SAM.pt"))
