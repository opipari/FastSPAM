import os
import json
import argparse
import torch

import numpy as np
from PIL import Image

from panopticapi.utils import id2rgb

from MVPd.utils.MVPdHelpers import label_to_one_hot
from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from video_panoptic_segmentation.metrics import utils as metric_utils





def rle_2_rgb(in_rle_dir, out_rgb_dir, dataset, device='cpu'):
    
    for video in dataset:
        sample = next(iter(video))
        video_name = sample['meta']['video_name']
        video_rle_dir = os.path.join(in_rle_dir, video_name)
        video_rgb_dir = os.path.join(out_rgb_dir, video_name)
        os.makedirs(video_rgb_dir, exist_ok=True)

        for sample in video:
            window_stamp = '.'.join(next(iter(sample['meta']['window_names'])).split('.')[:-1])
            rle_file = os.path.join(video_rle_dir, window_stamp+'.pt')
            rgb_file = os.path.join(video_rgb_dir, window_stamp+'.png')

            rle_segments = metric_utils.read_panomaskRLE(rle_file)
            rle_segments = rle_segments.to(dtype=torch.bool).to(device=device)

            ref_arr = sample['label']['mask'][0]
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
            rle_panomaskrgb.save(rgb_file)


        # video_rle_files = sorted(os.listdir(video_rle_dir))
        # # if os.path.isdir(video_rgb_dir):
        # #     continue
        # if video_name!="00808-y9hTuugGdiq.0000000000.0000000100":
        #     continue
        

        # panoptic_idgenerator = IdGenerator({el['id']: el for el in ref_dataset['categories']})
        # anno_meta = ref_dataset['annotations'][[anno['video_name'] for anno in ref_dataset['annotations']].index(video_name)]
        # video_class_dict = {int(key):int(value) for key,value in anno_meta['instance_id_to_category_id'].items()}
        # video_class_dict, to_merge_instance_id_to_stuff_id = merge_instances_of_stuff(panoptic_idgenerator,
        #                                                                                 stuff_category_ids,
        #                                                                                 video_class_dict)
        
        # for rle_file in video_rle_files:
        #     rgb_file = '.'.join(rle_file.split('.')[:-1])+'.png'

        #     rle_segments = metric_utils.read_panomaskRLE(os.path.join(video_rle_dir, rle_file))
        #     rle_segments = rle_segments.to(dtype=torch.bool).to(device=device)
            
        #     ref_arr = metric_utils.read_panomaskRGB(os.path.join(ref_rgb_dir, video_name, rgb_file))
        #     ref_arr = merge_stuff_labels(ref_arr, to_merge_instance_id_to_stuff_id)
        #     # Image.fromarray(id2rgb(ref_arr)).save(os.path.join(video_rgb_dir, '_'+rgb_file))
        #     ref_segments, ref_ids = label_to_one_hot(ref_arr, filter_void=True)
            
        #     ref_segments = torch.as_tensor(ref_segments, device=device, dtype=torch.bool)

        #     # Match rle segments to reference segments, then merge unmatched rle segments
        #     (matched_ref_ind, unmatched_ref_ind), (matched_rle_ind, unmatched_rle_ind), _ = metric_utils.match_segments(ref_segments, rle_segments)
            
        #     ref_rgbs = np.array(id2rgb(ref_ids))

        #     rle_rgbs = np.zeros((rle_segments.shape[0],4))
        #     # print(ref_rgbs[matched_ref_ind].shape, np.array(matched_ref_ind.shape[0]*[[255]]).shape)
        #     if ref_rgbs[matched_ref_ind].shape[0]>0:
        #         rle_rgbs[matched_rle_ind] = np.concatenate([ref_rgbs[matched_ref_ind], matched_ref_ind.shape[0]*[[255]]],axis=1)            
        #     rle_rgbs[unmatched_rle_ind] = (255,255,255,0.65*255)

        #     for unmatched_rle_i in unmatched_rle_ind:
        #         rle_segments[unmatched_rle_i] = torch.as_tensor(metric_utils.mask_to_boundary(rle_segments[unmatched_rle_i].cpu().float().numpy(), 
        #             dilation_ratio=0.004))

        #     rle_panomaskrgb = Image.fromarray(metric_utils.binmasks_to_panomask(rle_segments.cpu().numpy(), rle_rgbs).astype(np.uint8))
        #     rle_panomaskrgb.save(os.path.join(video_rgb_dir, rgb_file))
            



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rle_path',type=str, required=True) 
    parser.add_argument('--ref_path',type=str, required=True)
    parser.add_argument('--ref_split',type=str, required=True)
    args = parser.parse_args()
    

    in_rle_dir = os.path.join(args.rle_path, 'panomasksRLE')
    out_rgb_dir = os.path.join(args.rle_path, 'panomasksRGB')
    

    MVPd = MVPDataset(root=args.ref_path,
                        split=args.ref_split,
                        window_size = 0)
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rle_2_rgb(in_rle_dir, out_rgb_dir, MVPd, device=device)

