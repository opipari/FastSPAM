import os
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from scipy.optimize import linear_sum_assignment

import sys
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

#from zeroshot_rgbd.datasets.VaryingPerspectiveDataset import VaryingPerspectiveDataset
from zeroshot_rgbd.datasets.ActiveIlluminationDataset import ActiveIlluminationDataset
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
    res.save(os.path.join(dest_dir, str(prefix)+".SAM.jpg"))
    plt.close()

def show_anns(img, anns):
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
    plt.show()


def metrics(pred_img, label_img, eps=1e-10):

    num_gt = len(label_img)
    num_pred = len(pred_img)
    #print("num gt:", num_gt)
    #print("num pred:", num_pred)

    # confusion matrix (pred x gt)
    intersection_mat = np.zeros((num_pred, num_gt))
    precision_denom = np.sum(pred_img, axis=(1,2)).reshape(num_pred, 1)
    recall_denom = np.sum(label_img, axis=(1,2)).reshape(1, num_gt)
    #precision_mat = np.zeros((num_pred, num_gt))
    #recall_mat = np.zeros((num_pred, num_gt))


    for pred_idx in range(num_pred):
        pred_mask = pred_img[pred_idx]

        for gt_idx in range(num_gt):
            gt_mask = label_img[gt_idx]
            
            intersection_mat[pred_idx][gt_idx] = np.sum(gt_mask * pred_mask) + eps
            #precision_mat[pred_idx][gt_idx] = intersection / np.sum(pred_mask)
            #recall_mat[pred_idx][gt_idx] = intersection / np.sum(gt_mask)

    precision_mat = intersection_mat / precision_denom
    recall_mat = intersection_mat / recall_denom
    F_mat = (2*precision_mat*recall_mat) / (precision_mat + recall_mat)

    assignment = linear_sum_assignment(-F_mat)

    intersection_total = 0
    precision_denom_total = 0
    recall_denom_total = 0
    for pred_idx, gt_idx in zip(*assignment):
        intersection_total += intersection_mat[pred_idx][gt_idx]
        precision_denom_total += precision_denom[pred_idx][0]
        recall_denom_total += recall_denom[0][gt_idx]

    return intersection_total, precision_denom_total, recall_denom_total

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", dest="img_dir")
    parser.add_argument("--dest-dir", dest="dest_dir")
    parser.add_argument("--sam-checkpoint", dest="sam_checkpoint")
    parser.add_argument("--model-type", dest="model_type")
    parser.set_defaults(img_dir="./zeroshot_rgbd/datasets/renders/",
                    dest_dir="./zeroshot_rgbd/models/output/",
                    sam_checkpoint="./zeroshot_rgbd/models/segment_anything/checkpoints/sam_vit_h_4b8939.pth",
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


    for illumination in [None,5,10,25,50,100,200]:
        dest_dir="./zeroshot_rgbd/models/output_"+str(illumination)
        if not os.path.isdir(dest_dir):
            os.makedirs(dest_dir)
        dataset = ActiveIlluminationDataset(root_dir=img_dir, illumination=illumination)


        intersection_total = 0
        precision_denom_total = 0 
        recall_denom_total = 0

        for idx in range(len(dataset)):
            print("processing",idx)
            image, label = dataset[idx]
            image = image.permute(1,2,0).numpy()
            #label = label.permute(1,2,0).numpy()
            
            masks = mask_generator.generate(image)
            save_anns('.'.join(dataset.prefixes[idx]+[str(illumination)]), dest_dir, image, masks)

            # if len(masks)>0:
            #     pred = np.stack([ann['segmentation'] for ann in masks], axis=-1)
            # else:
            #     pred = np.ones(label.shape[:2]+(1,))
            
            # label = np.transpose(label, axes=(2,0,1))
            # pred = np.transpose(pred, axes=(2,0,1))

            # intersection, precision_denom, recall_denom = metrics(pred, label)
            
            # precision_img = intersection / precision_denom
            # recall_img = intersection / recall_denom
            # F_score_img = (2*precision_img*recall_img) / (precision_img + recall_img)
            # with open(os.path.join(dest_dir, dataset.prefixes[idx]+'.SAM.txt'), 'w') as f:
            #     f.write(f'Iter: {idx}/{len(dataset)}, Precision: {precision_img}, Recall: {recall_img}, F-Score: {F_score_img}\n')



            # intersection_total += intersection
            # precision_denom_total += precision_denom
            # recall_denom_total += recall_denom

            # if idx%10==0:
            #     precision = intersection_total / precision_denom_total
            #     recall = intersection_total / recall_denom_total
            #     F_score = (2*precision*recall) / (precision + recall)
            #     print(f'Iter: {idx}/{len(dataset)}, Precision: {precision}, Recall: {recall}, F-Score: {F_score}')
                

        # precision = intersection_total / precision_denom_total
        # recall = intersection_total / recall_denom_total
        # F_score = (2*precision*recall) / (precision + recall)
        # print(f'Final Result: Precision: {precision}, Recall: {recall}, F-Score: {F_score}')

        # with open(os.path.join(dest_dir,'final.SAM.txt'), 'w') as f:
        #     f.write(f'Final Result: Precision: {precision}, Recall: {recall}, F-Score: {F_score}\n')