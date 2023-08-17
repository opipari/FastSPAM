import os
import json
import argparse
import torch

import numpy as np
from PIL import Image

from panopticapi.utils import id2rgb

from MVPd.utils.MVPdHelpers import label_to_one_hot
from video_panoptic_segmentation.metrics import utils as metric_utils




def rle_2_rgb(in_rle_dir, out_rgb_dir, ref_rgb_dir, device='cpu'):

    for video_name in os.listdir(in_rle_dir):
        video_rle_dir = os.path.join(in_rle_dir, video_name)
        video_rle_files = sorted(os.listdir(video_rle_dir))

        video_rgb_dir = os.path.join(out_rgb_dir, video_name)
        os.makedirs(video_rgb_dir, exist_ok=True)
        
        for rle_file in video_rle_files:
            rgb_file = '.'.join(rle_file.split('.')[:-1])+'.png'

            rle_segments = metric_utils.read_panomaskRLE(os.path.join(video_rle_dir, rle_file))
            rle_segments = rle_segments.to(dtype=torch.bool).to(device=device)
            
            ref_arr = metric_utils.read_panomaskRGB(os.path.join(ref_rgb_dir, video_name, rgb_file))
            ref_segments, ref_ids = label_to_one_hot(ref_arr, filter_void=True)
            
            ref_segments = torch.as_tensor(ref_segments, device=device, dtype=torch.bool)

            # Match rle segments to reference segments, then merge unmatched rle segments
            (matched_ref_ind, unmatched_ref_ind, _), _, rle_segments_merged = metric_utils.match_and_merge_segments(ref_segments, rle_segments)
            
            ref_rgbs = np.array(id2rgb(ref_ids))
            rle_rgbs_merged = ref_rgbs[matched_ref_ind]
            
            rle_panomaskrgb = Image.fromarray(metric_utils.binmasks_to_panomask(rle_segments_merged.cpu().numpy(), rle_rgbs_merged).astype(np.uint8))
            rle_panomaskrgb.save(os.path.join(video_rgb_dir, rgb_file))
            



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rle_path',type=str, required=True)
    parser.add_argument('--ref_path',type=str, required=True)
    args = parser.parse_args()
    

    in_rle_dir = os.path.join(args.rle_path, 'panomasksRLE')
    out_rgb_dir = os.path.join(args.rle_path, 'panomasksRGB')
    ref_rgb_dir = os.path.join(args.ref_path, 'panomasksRGB')
    

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rle_2_rgb(in_rle_dir, out_rgb_dir, ref_rgb_dir, device=device)

