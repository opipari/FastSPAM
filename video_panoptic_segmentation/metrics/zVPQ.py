import os
import json
import argparse
import torch

import numpy as np
from PIL import Image

from MVPd.utils.MVPdHelpers import label_to_one_hot, merge_instances_of_stuff, merge_stuff_labels
from video_panoptic_segmentation.metrics import utils as metric_utils

from panopticapi.utils import IdGenerator, id2rgb





for k in [0,5,10,15]:


def get_video_annotations(video_name, dataset):
	dataset_video_names = [anno['video_name'] for anno in dataset['annotations']]
	assert video_name in dataset_video_names, f"Video, '{video_name}' not in dataset annotations"
	return dataset['annotations'][dataset_video_names.index(video_name)]

def zVPQ(true_rgb_dir, true_dataset, pred_rle_dir, device='cpu'):
    
    stuff_category_ids = [el['id'] for el in ref_dataset['categories'] if not el['isthing']]

    for video_name in os.listdir(in_rle_dir):
        video_rle_dir = os.path.join(in_rle_dir, video_name)
        video_rle_files = sorted(os.listdir(video_rle_dir))

        video_rgb_dir = os.path.join(true_rgb_dir, video_name)

        panoptic_idgenerator = IdGenerator({el['id']: el for el in ref_dataset['categories']})
        anno_meta = get_video_annotations(video_name, ref_dataset)
        video_class_dict = {int(key):int(value) for key,value in anno_meta['instance_id_to_category_id'].items()}
        video_class_dict, to_merge_instance_id_to_stuff_id = merge_instances_of_stuff(panoptic_idgenerator,
                                                                                        stuff_category_ids,
                                                                                        video_class_dict)
        
        for rle_file in video_rle_files:
            rgb_file = '.'.join(rle_file.split('.')[:-1])+'.png'

            rle_segments = metric_utils.read_panomaskRLE(os.path.join(video_rle_dir, rle_file))
            rle_segments = rle_segments.to(dtype=torch.bool).to(device=device)
            
            ref_arr = metric_utils.read_panomaskRGB(os.path.join(video_rgb_dir, rgb_file))
            ref_arr = merge_stuff_labels(ref_arr, to_merge_instance_id_to_stuff_id)
            ref_segments, ref_ids = label_to_one_hot(ref_arr, filter_void=True)
            
            ref_segments = torch.as_tensor(ref_segments, device=device, dtype=torch.bool)

            # Match rle segments to reference segments, then merge unmatched rle segments
            (matched_ref_ind, unmatched_ref_ind), (matched_rle_ind, unmatched_rle_ind), _ = metric_utils.match_segments(ref_segments, rle_segments)
            
            ref_rgbs = np.array(id2rgb(ref_ids))

            rle_rgbs = np.zeros((rle_segments.shape[0],4))
            # print(ref_rgbs[matched_ref_ind].shape, np.array(matched_ref_ind.shape[0]*[[255]]).shape)
            if ref_rgbs[matched_ref_ind].shape[0]>0:
                rle_rgbs[matched_rle_ind] = np.concatenate([ref_rgbs[matched_ref_ind], matched_ref_ind.shape[0]*[[255]]],axis=1)            
            rle_rgbs[unmatched_rle_ind] = (255,255,255,0.65*255)

            for unmatched_rle_i in unmatched_rle_ind:
                rle_segments[unmatched_rle_i] = torch.as_tensor(metric_utils.mask_to_boundary(rle_segments[unmatched_rle_i].cpu().float().numpy(), 
                    dilation_ratio=0.004))

            rle_panomaskrgb = Image.fromarray(metric_utils.binmasks_to_panomask(rle_segments.cpu().numpy(), rle_rgbs).astype(np.uint8))
            rle_panomaskrgb.save(os.path.join(video_rgb_dir, rgb_file))
            



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rle_path',type=str, required=True) 
    parser.add_argument('--ref_path',type=str, required=True)
    args = parser.parse_args()
    

    in_rle_dir = os.path.join(args.rle_path, 'panomasksRLE')
    out_rgb_dir = os.path.join(args.rle_path, 'panomasksRGB')
    ref_rgb_dir = os.path.join(args.ref_path, 'panomasksRGB')
    
    panoptic_file = next(iter([file for file in os.listdir(args.ref_path) if file.endswith(".json")]))
    ref_dataset = json.load(open(os.path.join(args.ref_path, panoptic_file),"r"))

    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rle_2_rgb(in_rle_dir, out_rgb_dir, ref_rgb_dir, ref_dataset, device=device)