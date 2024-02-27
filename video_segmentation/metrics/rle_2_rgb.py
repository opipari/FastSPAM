import os
import json
import argparse
import torch

import numpy as np
from PIL import Image

from panopticapi.utils import id2rgb

from MVPd.utils.MVPdHelpers import label_to_one_hot
from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from video_segmentation.metrics import utils as metric_utils





def rle_2_rgb(in_rle_dir, out_rgb_dir, dataset, v_name=None, device='cpu'):
    
    for video in dataset:
        sample = next(iter(video))
        video_name = sample['meta']['video_name']
        if v_name is not None and video_name!=v_name:
            continue
        video_rle_dir = os.path.join(in_rle_dir, video_name)
        video_rgb_dir = os.path.join(out_rgb_dir, video_name)
        os.makedirs(video_rgb_dir, exist_ok=True)

        for sample in video:
            window_stamp = '.'.join(next(iter(sample['meta']['window_names'])).split('.')[:-1])
            rle_file = os.path.join(video_rle_dir, window_stamp+'.pt')
            rgb_file = os.path.join(video_rgb_dir, window_stamp+'.png')

            rle_segments = metric_utils.read_panomaskRLE(rle_file)
            if len(rle_segments)==0:
                rle_segments = torch.zeros((1,480,640))
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
            rle_rgbs[unmatched_rle_ind] = (255,49,49,0.70*255)

            for unmatched_rle_i in unmatched_rle_ind:
                rle_segments[unmatched_rle_i] = torch.as_tensor(metric_utils.mask_to_boundary(rle_segments[unmatched_rle_i].cpu().float().numpy(), 
                    dilation_ratio=0.004))

            rle_panomaskrgb = Image.fromarray(metric_utils.binmasks_to_panomask(rle_segments.cpu().numpy(), rle_rgbs).astype(np.uint8))
            rle_panomaskrgb.save(rgb_file)




if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rle_path',type=str, required=True) 
    parser.add_argument('--ref_path',type=str, required=True)
    parser.add_argument('--ref_split',type=str, required=True)
    parser.add_argument('--v_name', type=str, default=None)
    args = parser.parse_args()
    

    in_rle_dir = os.path.join(args.rle_path, 'panomasksRLE')
    out_rgb_dir = os.path.join(args.rle_path, 'panomasksRGB')
    

    MVPd = MVPDataset(root=args.ref_path,
                        split=args.ref_split,
                        use_stuff=False,
                        filter_pcnt=0,
                        window_size = 0)
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rle_2_rgb(in_rle_dir, out_rgb_dir, MVPd, v_name=args.v_name, device=device)

