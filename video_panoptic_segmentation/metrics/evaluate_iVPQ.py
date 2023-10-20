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

import matplotlib.pyplot as plt

from panopticapi.utils import id2rgb

from MVPd.utils.MVPdHelpers import label_to_one_hot, filter_binmask_area
from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from video_panoptic_segmentation.metrics import utils as metric_utils





def collect_ref_window(video, start_idx, window_size=15):
    masks = []
    names = []
    zero_shot = {}
    for i in range(window_size):
        sample = video[start_idx+i]
        for k,v in sample['meta']['zero_shot_dict'].items():
            if k in zero_shot:
                assert zero_shot[k] == v
            else:
                zero_shot[k] = v
        mask = sample['label']['mask'][0]
        masks.append(F.interpolate(torch.as_tensor(mask)[None,None,:].to(torch.float), 
            scale_factor=(0.5,0.5), 
            mode='nearest')[0,0].numpy().astype(mask.dtype))
        names.append(sample['meta']['window_names'][0])
    return masks, names, zero_shot


def collect_rle_window(video_rle_dir, names, default_size=(1,480,640)):
    rle_segments = []
    for name in names:
        rle_file = os.path.join(video_rle_dir, '.'.join(name.split('.')[:-1])+'.pt')
        rle_seg = metric_utils.read_panomaskRLE(rle_file)
        if len(rle_seg)==0:
            rle_seg = torch.zeros(default_size)
        rle_segments.append(F.interpolate(rle_seg[None,:].to(torch.float), 
            scale_factor=(0.5,0.5), 
            mode='nearest')[0].to(torch.bool))
    return rle_segments


def get_tubes(rle_segments, window_size, segment_ids=None, cache=None, device='cpu'):
    orig_size = rle_segments[0].shape[1:]
    if cache is None:
        start_idx = 0
        tubes = torch.zeros((0,0,)+orig_size).to(dtype=torch.bool).to(device=device) # Window x N x 480 x 640
        tube_ids = [] if segment_ids is not None else None
        prev_segments = []
        prev_segments_map = {}
    else:
        start_idx, tubes, tube_ids, prev_segments, prev_segments_map = cache


    
    for i in range(start_idx, window_size):
        curr_segments = rle_segments[i]
        if segment_ids is not None and tube_ids is not None:
            curr_ids = segment_ids[i]
        curr_segments_map = {}

        T, N, H, W = tubes.shape

        if len(curr_segments)==0:
            tubes = torch.cat([tubes, torch.zeros((1,N,H,W), dtype=torch.bool, device=device)], dim=0) # T+1 x N x H x W
            if tube_ids is not None:
                tube_ids = tube_ids
        elif len(prev_segments)==0:
            curr_segments = curr_segments.to(dtype=torch.bool).to(device=device) # n x H x W
            curr_segments_map = {curr_ind: N+curr_ind for curr_ind in range(len(curr_segments))}

            curr_tubes = torch.cat([torch.zeros((N,H,W), dtype=torch.bool, device=device), curr_segments], dim=0).unsqueeze(0) # 1 x N+n x H x W
            tubes = torch.cat([tubes, torch.zeros((T,len(curr_segments),H,W), dtype=torch.bool, device=device)], dim=1) # T x N+n x H x W
            tubes = torch.cat([tubes, curr_tubes], dim=0) # T+1 x N+n x H x W
            if tube_ids is not None:
                tube_ids = tube_ids + curr_ids
        else:
            curr_segments = curr_segments.to(dtype=torch.bool).to(device=device)

            # Match rle segments to reference segments, then merge unmatched rle segments
            (matched_prev_ind, unmatched_prev_ind), (matched_curr_ind, unmatched_curr_ind), _ = metric_utils.match_segments(prev_segments, curr_segments)

            tubes = torch.cat([tubes, torch.zeros((1,N,H,W), dtype=torch.bool, device=device)], dim=0) # T+1 x N x H x W
            for prev_ind, curr_ind in zip(matched_prev_ind, matched_curr_ind):
                tubes[i,prev_segments_map[prev_ind]] = curr_segments[curr_ind]
                curr_segments_map[curr_ind] = prev_segments_map[prev_ind]
                # if tube_ids is not None:
                #     assert tube_ids[prev_segments_map[prev_ind]]==curr_ids[curr_ind]

            for ind_, curr_ind in enumerate(unmatched_curr_ind):
                curr_segments_map[curr_ind] = N+ind_
                if tube_ids is not None:
                    tube_ids.append(curr_ids[curr_ind])

            tubes = torch.cat([tubes, torch.zeros((T+1,len(unmatched_curr_ind),H,W), dtype=torch.bool, device=device)], dim=1) # T+1 x N+n x H x W
            tubes[T,range(N,N+len(unmatched_curr_ind))] = curr_segments[unmatched_curr_ind] # T+1 x N+n x H x W

        prev_segments = curr_segments
        prev_segments_map = curr_segments_map

    return window_size, tubes, tube_ids, prev_segments, prev_segments_map


def evaluate_iVPQ_v2(in_rle_dir, out_dir, ref_path, ref_split, device='cpu', i=0, n_proc=0, k_list=[1,5,10,15], step_size=15):
    
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
        
        if os.path.exists(os.path.join(video_out_dir, "metrics_iVPQ_v2.json")):
            continue

        
        video_results = {
                        "zero-shot": {k: {"IOU": 0, "TP": 0, "FP": 0, "FN": 0} for k in k_list},
                        "non-zero-shot": {k: {"IOU": 0, "TP": 0, "FP": 0, "FN": 0} for k in k_list}
                        }
        window_size = max(k_list)
        

        for v_idx in tqdm(range(0, len(video)-window_size, step_size), position=1, disable=i!=0):

            ref_arr, ref_names, ref_zero_shot = collect_ref_window(video, start_idx=v_idx, window_size=window_size)
            ref_seg_id = [label_to_one_hot(arr, filter_void=True) for arr in ref_arr]
            ref_segments, ref_ids = [torch.as_tensor(d[0]) for d in ref_seg_id], [list(d[1]) for d in ref_seg_id]

            rle_segments = collect_rle_window(video_rle_dir, ref_names)

            for ki, k in enumerate(k_list):
                if ki==0:
                    ref_tube_cache = get_tubes(ref_segments, window_size=k, segment_ids=ref_ids, device=device)
                    rle_tube_cache = get_tubes(rle_segments, window_size=k, device=device)
                else:
                    ref_tube_cache = get_tubes(ref_segments, window_size=k, segment_ids=ref_ids, cache=ref_tube_cache, device=device)
                    rle_tube_cache = get_tubes(rle_segments, window_size=k, cache=rle_tube_cache, device=device)

                ref_tubes, ref_tubes_ids = ref_tube_cache[1:3]
                rle_tubes = rle_tube_cache[1]

                assert all(ref_id in ref_zero_shot for ref_id in ref_tubes_ids)
                
                rle_tubes = rle_tubes.permute(1,0,2,3)
                rle_tubes = torch.flatten(rle_tubes, start_dim=1,end_dim=2)

                
                ref_tubes = ref_tubes.permute(1,0,2,3)
                ref_tubes = torch.flatten(ref_tubes, start_dim=1,end_dim=2)

                

                (matched_ref_ind, unmatched_ref_ind), (matched_rle_ind, unmatched_rle_ind), _ = metric_utils.match_segments(ref_tubes, rle_tubes)                
                
                for ref_ind, rle_ind in zip(matched_ref_ind, matched_rle_ind):
                    anno_mask, pred_mask = ref_tubes[ref_ind], rle_tubes[rle_ind]

                    metrics = metric_utils.segment_metrics(anno_mask, pred_mask)
                    if ref_zero_shot[ref_tubes_ids[ref_ind]]:
                        video_results["zero-shot"][k]["IOU"] += metrics["IOU"]
                        video_results["zero-shot"][k]["TP"] += 1
                    else:
                        video_results["non-zero-shot"][k]["IOU"] += metrics["IOU"]
                        video_results["non-zero-shot"][k]["TP"] += 1

                for rle_ind in unmatched_rle_ind:
                    video_results["zero-shot"][k]["FP"] += 1
                    video_results["non-zero-shot"][k]["FP"] += 1
                    
                for ref_ind in unmatched_ref_ind:
                    if ref_zero_shot[ref_tubes_ids[ref_ind]]:
                        video_results["zero-shot"][k]["FN"] += 1
                    else:
                        video_results["non-zero-shot"][k]["FN"] += 1
                
        os.makedirs(video_out_dir, exist_ok=True)
        with open(os.path.join(video_out_dir, "metrics_iVPQ_v2.json"),"w") as fl:
            json.dump(video_results, fl)




def evaluate_iVPQ_v1(in_rle_dir, out_dir, ref_path, ref_split, device='cpu', i=0, n_proc=0, k_list=[1,5,10,15], step_size=15):
    
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
        
        if os.path.exists(os.path.join(video_out_dir, "metrics_iVPQ_v1.json")):
            continue

        
        video_results = {
                        "zero-shot": {k: {"IOU": 0, "TP": 0, "FP": 0, "FN": 0} for k in k_list},
                        "non-zero-shot": {k: {"IOU": 0, "TP": 0, "FP": 0, "FN": 0} for k in k_list}
                        }
        window_size = max(k_list)
        

        for v_idx in tqdm(range(0, len(video)-window_size, step_size), position=1, disable=i!=0):

            ref_arr, ref_names, ref_zero_shot = collect_ref_window(video, start_idx=v_idx, window_size=window_size)
            ref_segments, ref_ids = label_to_one_hot(np.stack(ref_arr), filter_void=True)

            rle_segments = collect_rle_window(video_rle_dir, ref_names)


            for ki, k in enumerate(k_list):
                if ki==0:
                    rle_tube_cache = get_tubes(rle_segments, window_size=k, device=device)
                else:
                    rle_tube_cache = get_tubes(rle_segments, window_size=k, cache=rle_tube_cache, device=device)
                
                ref_tubes, ref_tubes_ids = ref_segments[:k], ref_ids
                non_empty_ref = np.sum(ref_tubes, axis=(0,2,3))>0
                ref_tubes, ref_tubes_ids = ref_tubes[:,non_empty_ref], ref_tubes_ids[non_empty_ref]
                ref_tubes = torch.as_tensor(ref_tubes, dtype=torch.bool, device=device)
                
                rle_tubes = rle_tube_cache[1]

                assert all(ref_id in ref_zero_shot for ref_id in ref_tubes_ids)
                
                rle_tubes = rle_tubes.permute(1,0,2,3)
                rle_tubes = torch.flatten(rle_tubes, start_dim=1,end_dim=2)

                
                ref_tubes = ref_tubes.permute(1,0,2,3)
                ref_tubes = torch.flatten(ref_tubes, start_dim=1,end_dim=2)

                
                (matched_ref_ind, unmatched_ref_ind), (matched_rle_ind, unmatched_rle_ind), _ = metric_utils.match_segments(ref_tubes, rle_tubes)                

                for ref_ind, rle_ind in zip(matched_ref_ind, matched_rle_ind):
                    anno_mask, pred_mask = ref_tubes[ref_ind], rle_tubes[rle_ind]

                    metrics = metric_utils.segment_metrics(anno_mask, pred_mask)
                    if ref_zero_shot[ref_tubes_ids[ref_ind]]:
                        video_results["zero-shot"][k]["IOU"] += metrics["IOU"]
                        video_results["zero-shot"][k]["TP"] += 1
                    else:
                        video_results["non-zero-shot"][k]["IOU"] += metrics["IOU"]
                        video_results["non-zero-shot"][k]["TP"] += 1

                for rle_ind in unmatched_rle_ind:
                    video_results["zero-shot"][k]["FP"] += 1
                    video_results["non-zero-shot"][k]["FP"] += 1
                    
                for ref_ind in unmatched_ref_ind:
                    if ref_zero_shot[ref_tubes_ids[ref_ind]]:
                        video_results["zero-shot"][k]["FN"] += 1
                    else:
                        video_results["non-zero-shot"][k]["FN"] += 1

        os.makedirs(video_out_dir, exist_ok=True)
        with open(os.path.join(video_out_dir, "metrics_iVPQ_v1.json"),"w") as fl:
            json.dump(video_results, fl)




def summarize_iVPQ(metric_root, k_list=['1','5','10','15'], fpath="metrics_iVPQ.json"):
    total = {
            "zero-shot": {k: {"sum_iou": 0, "sum_denom": 0} for k in k_list},
            "non-zero-shot": {k: {"sum_iou": 0, "sum_denom": 0} for k in k_list}
            }
    v_tot = 0
    for video_name in os.listdir(metric_root):
        video_data = json.load(open(os.path.join(metric_root, video_name, fpath),"r"))
        for zs in ["zero-shot", "non-zero-shot"]:
            for k in k_list:
                if (video_data[zs][k]["TP"] + video_data[zs][k]["FN"]) > 0:
                    sum_iou = video_data[zs][k]["IOU"]
                    denom = video_data[zs][k]["TP"] + (0.5*video_data[zs][k]["FP"]) + (0.5*video_data[zs][k]["FN"])
                else:
                    sum_iou = 0
                    denom = 0
                total[zs][k]["sum_iou"] += sum_iou
                total[zs][k]["sum_denom"] += denom
                print(zs, video_name, k, sum_iou/denom if denom>0 else 0)
        v_tot+=1
        print()
    print(v_tot)
    for zs in ["zero-shot", "non-zero-shot"]:
        avg=0
        for k in k_list:
            ivpq_k = total[zs][k]["sum_iou"]/total[zs][k]["sum_denom"]
            avg+=ivpq_k
            print(zs, "Total", k, ivpq_k)
        print(zs,"Total", "iVPQ", avg/len(k_list))





if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rle_path',type=str, required=True) 
    parser.add_argument('--ref_path',type=str, required=True)
    parser.add_argument('--ref_split',type=str, required=True)
    parser.add_argument('--compute', action='store_true', default=False)
    parser.add_argument('--summarize', action='store_true', default=False)
    parser.add_argument('--n_proc', type=int, default=1)
    args = parser.parse_args()
    

    in_rle_dir = os.path.join(args.rle_path, 'panomasksRLE')
    out_json_dir = os.path.join(args.rle_path, 'metricsJSON')
    


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # rle_2_rgb(in_rle_dir, out_rgb_dir, MVPd, device=device)

    # n_proc = args.n_proc
    # mp.set_start_method('spawn', force=True)
    # pool = mp.Pool(processes = n_proc)
    # pool.starmap(evaluate_iVPQ, [[in_rle_dir, out_json_dir, args.ref_path, args.ref_split, device, i, n_proc] for i in range(n_proc)])

    if args.compute:
        evaluate_iVPQ_v1(in_rle_dir, out_json_dir, args.ref_path, args.ref_split, device, 0, 0)
    if args.summarize:
        summarize_iVPQ(out_json_dir, fpath="metrics_iVPQ_v1.json")
