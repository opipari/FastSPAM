import os
import json
import math
import argparse

from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import time



from panopticapi.utils import id2rgb

from MVPd.utils.MVPdHelpers import label_to_one_hot, filter_binmask_area
from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from video_panoptic_segmentation.metrics import utils as metric_utils




def get_mask_boundary_metrics(anno_mask, pred_mask, device='cpu'):
    anno_boundary = metric_utils.mask_to_boundary(anno_mask.cpu().numpy().astype(np.uint8), dilation_ratio=0.002)
    pred_boundary = metric_utils.mask_to_boundary(pred_mask.cpu().numpy().astype(np.uint8), dilation_ratio=0.002)
    anno_boundary = torch.as_tensor(anno_boundary > 0, device=device)
    pred_boundary = torch.as_tensor(pred_boundary > 0, device=device)

    mask_metrics = {"Mask-"+key: value for key,value in metric_utils.segment_metrics(anno_mask, pred_mask).items()}
    boundary_metrics = {"Boundary-"+key: value for key,value in metric_utils.segment_metrics(anno_boundary, pred_boundary).items()}

    return mask_metrics, boundary_metrics


def collect_ref_window(video, start_idx, window_size=15):
    masks = []
    names = []
    for i in range(window_size):
        sample = video[start_idx+i]
        mask = sample['label']['mask'][0]
        masks.append(F.interpolate(torch.as_tensor(mask).unsqueeze(0).to(torch.float), 
            scale_factor=0.5, 
            mode='nearest')[0].numpy().astype(mask.dtype))
        names.append(sample['meta']['window_names'][0])
    return np.array(masks), names


def collect_rle_window(video_rle_dir, names):
    rle_segments = []
    for name in names:
        rle_file = os.path.join(video_rle_dir, '.'.join(name.split('.')[:-1])+'.pt')
        rle_segments.append(F.interpolate(metric_utils.read_panomaskRLE(rle_file).to(torch.float), 
            scale_factor=0.5, 
            mode='nearest').to(torch.bool))
    return rle_segments


def get_rle_tubes(rle_segments, window_size, device='cpu'):
    orig_size = rle_segments[0].shape[1:]
    # rle_segments = [F.interpolate(seg.to(torch.float), scale_factor=0.5, mode='nearest').to(torch.bool) for seg in rle_segments]
    # intr_size = rle_segments[0].shape[1:]
    tubes = torch.zeros((0,0,)+orig_size).to(dtype=torch.bool).to(device=device) # Window x N x 480 x 640

    prev_segments = []
    prev_segments_map = {}
    for i in range(window_size):
        curr_segments = rle_segments[i]
        curr_segments_map = {}

        T, N, H, W = tubes.shape

        if len(curr_segments)==0:
            tubes = torch.cat([tubes, torch.zeros((1,N,H,W), dtype=torch.bool, device=device)], dim=0) # T+1 x N x H x W
        elif len(prev_segments)==0:
            curr_segments = curr_segments.to(dtype=torch.bool).to(device=device) # n x H x W
            curr_segments_map = {curr_ind: N+curr_ind for curr_ind in range(len(curr_segments))}

            curr_tubes = torch.cat([torch.zeros((N,H,W), dtype=torch.bool, device=device), curr_segments], dim=0).unsqueeze(0) # 1 x N+n x H x W
            tubes = torch.cat([tubes, torch.zeros((T,len(curr_segments),H,W), dtype=torch.bool, device=device)], dim=1) # T x N+n x H x W
            tubes = torch.cat([tubes, curr_tubes], dim=0) # T+1 x N+n x H x W
        else:
            curr_segments = curr_segments.to(dtype=torch.bool).to(device=device)

            # Match rle segments to reference segments, then merge unmatched rle segments
            (matched_prev_ind, unmatched_prev_ind), (matched_curr_ind, unmatched_curr_ind), _ = metric_utils.match_segments(prev_segments, curr_segments)

            tubes = torch.cat([tubes, torch.zeros((1,N,H,W), dtype=torch.bool, device=device)], dim=0) # T+1 x N x H x W
            for prev_ind, curr_ind in zip(matched_prev_ind, matched_curr_ind):
                tubes[i,prev_segments_map[prev_ind]] = curr_segments[curr_ind]
                curr_segments_map[curr_ind] = prev_segments_map[prev_ind]

            for ind_, curr_ind in enumerate(unmatched_curr_ind):
                curr_segments_map[curr_ind] = N+ind_

            tubes = torch.cat([tubes, torch.zeros((T+1,len(unmatched_curr_ind),H,W), dtype=torch.bool, device=device)], dim=1) # T+1 x N+n x H x W
            tubes[T,range(N,N+len(unmatched_curr_ind))] = curr_segments[unmatched_curr_ind] # T+1 x N+n x H x W

        prev_segments = curr_segments
        prev_segments_map = curr_segments_map

    return tubes #F.interpolate(tubes.to(torch.float), size=orig_size, mode='nearest').to(torch.bool)


def evaluate_metrics(in_rle_dir, out_dir, ref_path, ref_split, device='cpu', i=0, n_proc=0):
    
    dataset = MVPDataset(root=ref_path,
                        split=ref_split,
                        window_size = 0)
    if n_proc>0:
        is_per_proc = math.ceil(len(dataset)/n_proc)
        i_start = i*is_per_proc
        i_end = min((i+1)*is_per_proc, len(dataset))
        inds = list(range(i_start, i_end))
        dataset = torch.utils.data.Subset(dataset, inds)
        print(len(dataset))
        
    for video in tqdm(dataset, position=0, disable=i!=0):
        sample = next(iter(video))
        video_name = sample['meta']['video_name']
        video_rle_dir = os.path.join(in_rle_dir, video_name)
        video_out_dir = os.path.join(out_dir, video_name)
        os.makedirs(video_out_dir, exist_ok=True)
        
        if os.path.exists(os.path.join(video_out_dir, "metrics.json")):
            continue

        import time

        k_list=[1,5,10,15]
        video_results = {k: {"IOU":[], "TP": 0, "FP": 0, "FN": 0} for k in k_list}
        window_size = max(k_list)
        ref_arr, ref_names = collect_ref_window(video, start_idx=0, window_size=window_size)
        rle_segments = collect_rle_window(video_rle_dir, ref_names)

        for v_idx in tqdm(range(window_size, len(video)), position=1, disable=i!=0):
            sample = video[v_idx]
            ref_arr[:-1] = ref_arr[1:]
            ref_arr[-1] = F.interpolate(torch.as_tensor(sample['label']['mask'][0]).unsqueeze(0).to(torch.float), 
                scale_factor=0.5, 
                mode='nearest')[0].numpy().astype(ref_arr.dtype)
            
            rle_file = os.path.join(video_rle_dir, '.'.join(sample['meta']['window_names'][0].split('.')[:-1])+'.pt')
            rle_segments = rle_segments[1:]
            rle_segments.append(F.interpolate(metric_utils.read_panomaskRLE(rle_file).to(torch.float), 
                scale_factor=0.5, 
                mode='nearest').to(torch.bool))
            
            rle_tubes = get_rle_tubes(rle_segments, window_size, device=device)

            ref_segments, ref_ids = label_to_one_hot(ref_arr, filter_void=True)
            ref_segments, ref_inds = filter_binmask_area(ref_segments)
            ref_ids = ref_ids[ref_inds]
            ref_tubes = torch.as_tensor(ref_segments, device=device, dtype=torch.bool)
            # ref_tubes = get_rle_tubes([torch.as_tensor(ar, device=device, dtype=torch.bool) for ar in ref_segments], window_size, default_size=(480,640), device=device)


            for k in k_list:
                rle_tubes_ = rle_tubes[:k]
                rle_tubes_ = rle_tubes_[:,torch.any(torch.sum(rle_tubes_, dim=(-2,-1)), dim=0)]
                rle_tubes_ = rle_tubes_.permute(1,0,2,3)
                rle_tubes_ = torch.flatten(rle_tubes_, start_dim=1,end_dim=2)

                ref_tubes_ = ref_tubes[:k]
                ref_tubes_ = ref_tubes_.permute(1,0,2,3)
                ref_tubes_ = torch.flatten(ref_tubes_, start_dim=1,end_dim=2)

                (matched_ref_ind, unmatched_ref_ind), (matched_rle_ind, unmatched_rle_ind), _ = metric_utils.match_segments(ref_tubes_, rle_tubes_)                

                for ref_ind, rle_ind in zip(matched_ref_ind, matched_rle_ind):
                    anno_mask, pred_mask = ref_tubes_[ref_ind], rle_tubes_[rle_ind]

                    metrics = metric_utils.segment_metrics(anno_mask, pred_mask)
                    video_results[k]["IOU"].append(metrics["IOU"])
                    video_results[k]["TP"] += 1

                for rle_ind in unmatched_rle_ind:
                    video_results[k]["FP"] += 1
                    
                for ref_ind in unmatched_ref_ind:
                    video_results[k]["FN"] += 1

        with open(os.path.join(video_out_dir, "metrics_iVPQ.json"),"w") as fl:
            json.dump(video_results, fl)





            




if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rle_path',type=str, required=True) 
    parser.add_argument('--ref_path',type=str, required=True)
    parser.add_argument('--ref_split',type=str, required=True)
    parser.add_argument('--n_proc', type=int, default=1)
    args = parser.parse_args()
    

    in_rle_dir = os.path.join(args.rle_path, 'panomasksRLE')
    out_rgb_dir = os.path.join(args.rle_path, 'metricsJSON')
    


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # rle_2_rgb(in_rle_dir, out_rgb_dir, MVPd, device=device)

    # n_proc = args.n_proc
    # mp.set_start_method('spawn', force=True)
    # pool = mp.Pool(processes = n_proc)
    # pool.starmap(evaluate_metrics, [[in_rle_dir, out_rgb_dir, args.ref_path, args.ref_split, device, i, n_proc] for i in range(n_proc)])

    evaluate_metrics(in_rle_dir, out_rgb_dir, args.ref_path, args.ref_split, device, 0, 0)

