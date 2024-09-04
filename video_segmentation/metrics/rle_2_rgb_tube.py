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



def collect_rle_tracks(video, video_rle_dir, threshold=0.5):
    track_inds = set()

    for f_idx in range(len(video)):
        # if f_idx>49:
        #     break
        rle_file = os.path.join(video_rle_dir, f'{f_idx:010d}.pt')
        track_boxes = torch.load(rle_file, map_location='cpu')['bbox_results']
        track_inds.update(np.flatnonzero(track_boxes[:,5]>threshold))

    return list(track_inds)

def rle_2_rgb(in_rle_dir, out_rgb_dir, dataset, v_name=None, device='cpu'):
    
    for video in dataset:
        sample = next(iter(video))
        video_name = sample['meta']['video_name']
        if v_name is not None and video_name!=v_name:
            continue
        video_rle_dir = os.path.join(in_rle_dir, video_name)
        video_rgb_dir = os.path.join(out_rgb_dir, video_name)
        os.makedirs(video_rgb_dir, exist_ok=True)

        track_inds = collect_rle_tracks(video, video_rle_dir)
        print(track_inds)
        for sample in video:
            window_stamp = '.'.join(next(iter(sample['meta']['window_names'])).split('.')[:-1])
            rle_file = os.path.join(video_rle_dir, window_stamp+'.pt')
            rgb_file = os.path.join(video_rgb_dir, window_stamp+'.png')

            rle_segments = metric_utils.read_panomaskRLE(rle_file, track_inds)
            if len(rle_segments)==0:
                rle_segments = torch.zeros((len(track_inds),480,640))
            rle_segments = rle_segments.to(dtype=torch.bool).to(device=device)

            
            cmap = metric_utils.get_100_cmap()
            rle_rgbs = np.array([cmap(99-i) for i in range(len(track_inds))])*255
            

            

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

