import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image

import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def save_anns(index, dest_dir, img, anns):
    fig = plt.figure(figsize=(20,20))
    plt.axis('off')
    plt.imshow(image)

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
    plt.savefig(os.path.join(dest_dir, str(index)+"_SAM.png"))
    Image.frombytes('RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", dest="img_dir")
    parser.add_argument("--dest-dir", dest="dest_dir")
    parser.add_argument("--sam-checkpoint", dest="sam_checkpoint")
    parser.add_argument("--model-type", dest="model_type")
    parser.set_defaults(img_dir="/media/mytre/0CD418EB76995EEF/SegmentationProject/simulators/test/",
                    dest_dir="/media/mytre/0CD418EB76995EEF/SegmentationProject/simulators/test/",
                    sam_checkpoint="/media/mytre/0CD418EB76995EEF/SegmentationProject/models/segment_anything/checkpoints/sam_vit_h_4b8939.pth",
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



    for fl in os.listdir(img_dir):
        if fl.endswith("_rgb.png"):
            index = fl.split("_")[0]
            image = cv2.imread(os.path.join(img_dir, fl))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masks = mask_generator.generate(image)
            save_anns(index, dest_dir, image, masks)
            

            

            

            