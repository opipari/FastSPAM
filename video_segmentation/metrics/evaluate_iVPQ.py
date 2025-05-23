import os
import json
import math
import argparse
import pickle

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
from video_segmentation.metrics import utils as metric_utils






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


def collect_rle_window(video_rle_dir, names, inds=None, default_size=(1,480,640)):
    rle_segments = []
    for name in names:
        rle_file = os.path.join(video_rle_dir, '.'.join(name.split('.')[:-1])+'.pt')
        rle_seg = metric_utils.read_panomaskRLE(rle_file, inds=inds)
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



def evaluate_iVPQ_v1(in_rle_dir, out_dir, dataset, device='cpu', i=0, n_proc=0, k_list=[1,5,10,15], step_size=15):
    
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



def collect_rle_tracks(video, video_rle_dir, threshold=0.5):
    track_inds = set()

    for f_idx in range(len(os.listdir(video_rle_dir))):
        rle_file = os.path.join(video_rle_dir, f'{f_idx:010d}.pt')
        track_boxes = torch.load(rle_file, map_location='cpu')['bbox_results']
        track_inds.update(np.flatnonzero(track_boxes[:,5]>threshold))

    return list(track_inds)



def evaluate_iVPQ_vkn(in_rle_dir, out_dir, dataset, device='cpu', i=0, n_proc=0, k_list=[1,5,10,15], step_size=15, model_type='swin'):
    
    if n_proc>0:
        is_per_proc = math.ceil(len(dataset)/n_proc)
        i_start = i*is_per_proc
        i_end = min((i+1)*is_per_proc, len(dataset))
        inds = list(range(i_start, i_end))
        dataset = torch.utils.data.Subset(dataset, inds)
        print(len(dataset))
        
    for vii,video in enumerate(tqdm(dataset, position=0, disable=i!=0)):
        if len(video)>300:
            continue
        sample = next(iter(video))
        video_name = sample['meta']['video_name']
        video_rle_dir = os.path.join(in_rle_dir, video_name)
        video_out_dir = os.path.join(out_dir, video_name)
        print(vii,len(dataset))
        
        if os.path.exists(os.path.join(video_out_dir, "metrics_iVPQ_v1.json")):
            continue

        
        video_results = {
                        "zero-shot": {k: {"IOU": 0, "TP": 0, "FP": 0, "FN": 0} for k in k_list},
                        "non-zero-shot": {k: {"IOU": 0, "TP": 0, "FP": 0, "FN": 0} for k in k_list}
                        }
        window_size = max(k_list)
        
        rle_track_inds = collect_rle_tracks(video, video_rle_dir, threshold=0.5)

        if model_type=='r50':
            video_length = len(video)
        else:
            video_length = len(video) - (len(video) % 64)

        for v_idx in tqdm(range(0, video_length-window_size, step_size), position=1, disable=i!=0):

            ref_arr, ref_names, ref_zero_shot = collect_ref_window(video, start_idx=v_idx, window_size=window_size)
            ref_segments, ref_ids = label_to_one_hot(np.stack(ref_arr), filter_void=True)

            # tt = time.time()
            rle_segments = collect_rle_window(video_rle_dir, ref_names, inds=rle_track_inds, default_size=(100,480,640))
            # print(time.time()-tt)
            rle_segments = torch.stack(rle_segments).to(device=device)
            for ki, k in enumerate(k_list):
                # if ki==0:
                #     rle_tube_cache = get_tubes(rle_segments, window_size=k, device=device)
                # else:
                #     rle_tube_cache = get_tubes(rle_segments, window_size=k, cache=rle_tube_cache, device=device)
                
                ref_tubes, ref_tubes_ids = ref_segments[:k], ref_ids
                non_empty_ref = np.sum(ref_tubes, axis=(0,2,3))>0
                ref_tubes, ref_tubes_ids = ref_tubes[:,non_empty_ref], ref_tubes_ids[non_empty_ref]
                ref_tubes = torch.as_tensor(ref_tubes, dtype=torch.bool, device=device)
                
                rle_tubes = rle_segments[:k] #rle_tube_cache[1]

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
            "non-zero-shot": {k: {"sum_iou": 0, "sum_denom": 0} for k in k_list},
            "aggregate": {k: {"sum_iou": 0, "sum_denom": 0} for k in k_list}
            }
    v_tot = 0
    # for video_name in sorted(os.listdir(metric_root)):
    for video_name in ['00815-h1zeeAwLh9Z.0000000033.0000001000', '00827-BAbdmeyTvMZ.0000000029.0000001000', '00839-zt1RVoi7PcG.0000000025.0000000100', '00848-ziup5kvtCCR.0000000008.0000000100', '00827-BAbdmeyTvMZ.0000000021.0000001000', '00835-q3zU7Yy5E5s.0000000033.0000001000', '00835-q3zU7Yy5E5s.0000000029.0000000100', '00844-q5QZSEeHe5g.0000000023.0000001000', '00832-qyAac8rV8Zk.0000000036.0000001000', '00831-yr17PDCnDDW.0000000048.0000000100', '00815-h1zeeAwLh9Z.0000000014.0000000100', '00827-BAbdmeyTvMZ.0000000046.0000001000', '00891-cvZr5TUy5C5.0000000000.0000000100', '00844-q5QZSEeHe5g.0000000000.0000001000', '00835-q3zU7Yy5E5s.0000000015.0000000100', '00843-DYehNKdT76V.0000000027.0000000100', '00849-a8BtkwhxdRV.0000000012.0000001000', '00849-a8BtkwhxdRV.0000000007.0000000100', '00832-qyAac8rV8Zk.0000000008.0000001000', '00871-VBzV5z6i1WS.0000000002.0000001000', '00831-yr17PDCnDDW.0000000014.0000000100', '00873-bxsVRursffK.0000000005.0000000100', '00871-VBzV5z6i1WS.0000000045.0000000100', '00871-VBzV5z6i1WS.0000000040.0000000100', '00844-q5QZSEeHe5g.0000000048.0000000100', '00823-7MXmsvcQjpJ.0000000026.0000000100', '00848-ziup5kvtCCR.0000000011.0000000100', '00831-yr17PDCnDDW.0000000013.0000000100', '00843-DYehNKdT76V.0000000016.0000001000', '00808-y9hTuugGdiq.0000000036.0000001000', '00843-DYehNKdT76V.0000000022.0000001000', '00891-cvZr5TUy5C5.0000000026.0000000100', '00878-XB4GS9ShBRE.0000000029.0000000100', '00849-a8BtkwhxdRV.0000000033.0000001000', '00878-XB4GS9ShBRE.0000000003.0000000100', '00843-DYehNKdT76V.0000000031.0000000100', '00823-7MXmsvcQjpJ.0000000030.0000000100', '00832-qyAac8rV8Zk.0000000000.0000000100', '00878-XB4GS9ShBRE.0000000039.0000001000', '00810-CrMo8WxCyVb.0000000001.0000000100', '00832-qyAac8rV8Zk.0000000031.0000001000', '00848-ziup5kvtCCR.0000000000.0000000100', '00843-DYehNKdT76V.0000000011.0000001000', '00878-XB4GS9ShBRE.0000000028.0000000100', '00848-ziup5kvtCCR.0000000003.0000001000', '00832-qyAac8rV8Zk.0000000024.0000000100', '00815-h1zeeAwLh9Z.0000000030.0000001000', '00832-qyAac8rV8Zk.0000000015.0000001000', '00808-y9hTuugGdiq.0000000030.0000000100', '00843-DYehNKdT76V.0000000049.0000000100', '00808-y9hTuugGdiq.0000000007.0000001000', '00848-ziup5kvtCCR.0000000006.0000000100', '00839-zt1RVoi7PcG.0000000009.0000001000', '00849-a8BtkwhxdRV.0000000035.0000000100', '00891-cvZr5TUy5C5.0000000028.0000000100', '00813-svBbv1Pavdk.0000000003.0000000100', '00831-yr17PDCnDDW.0000000038.0000001000', '00843-DYehNKdT76V.0000000009.0000000100', '00823-7MXmsvcQjpJ.0000000018.0000000100', '00810-CrMo8WxCyVb.0000000046.0000001000', '00831-yr17PDCnDDW.0000000004.0000001000', '00832-qyAac8rV8Zk.0000000027.0000001000', '00871-VBzV5z6i1WS.0000000029.0000000100', '00891-cvZr5TUy5C5.0000000000.0000001000', '00808-y9hTuugGdiq.0000000011.0000001000', '00827-BAbdmeyTvMZ.0000000023.0000001000', '00878-XB4GS9ShBRE.0000000031.0000000100', '00891-cvZr5TUy5C5.0000000016.0000001000', '00823-7MXmsvcQjpJ.0000000048.0000000100', '00813-svBbv1Pavdk.0000000004.0000000100', '00827-BAbdmeyTvMZ.0000000002.0000001000', '00815-h1zeeAwLh9Z.0000000043.0000001000', '00848-ziup5kvtCCR.0000000045.0000001000', '00810-CrMo8WxCyVb.0000000003.0000001000', '00843-DYehNKdT76V.0000000039.0000000100', '00832-qyAac8rV8Zk.0000000007.0000000100', '00878-XB4GS9ShBRE.0000000020.0000000100', '00832-qyAac8rV8Zk.0000000015.0000000100', '00848-ziup5kvtCCR.0000000001.0000001000', '00871-VBzV5z6i1WS.0000000032.0000000100', '00813-svBbv1Pavdk.0000000007.0000001000', '00835-q3zU7Yy5E5s.0000000006.0000001000', '00849-a8BtkwhxdRV.0000000037.0000001000', '00831-yr17PDCnDDW.0000000034.0000001000', '00849-a8BtkwhxdRV.0000000044.0000000100', '00878-XB4GS9ShBRE.0000000008.0000001000', '00823-7MXmsvcQjpJ.0000000012.0000000100', '00831-yr17PDCnDDW.0000000014.0000001000', '00891-cvZr5TUy5C5.0000000033.0000001000', '00843-DYehNKdT76V.0000000030.0000001000', '00873-bxsVRursffK.0000000036.0000000100', '00823-7MXmsvcQjpJ.0000000012.0000001000', '00839-zt1RVoi7PcG.0000000000.0000000100', '00832-qyAac8rV8Zk.0000000030.0000000100', '00873-bxsVRursffK.0000000024.0000001000', '00832-qyAac8rV8Zk.0000000031.0000000100', '00873-bxsVRursffK.0000000019.0000000100', '00844-q5QZSEeHe5g.0000000015.0000001000', '00891-cvZr5TUy5C5.0000000016.0000000100', '00827-BAbdmeyTvMZ.0000000034.0000001000', '00849-a8BtkwhxdRV.0000000010.0000001000', '00808-y9hTuugGdiq.0000000049.0000001000', '00891-cvZr5TUy5C5.0000000040.0000001000', '00831-yr17PDCnDDW.0000000038.0000000100', '00871-VBzV5z6i1WS.0000000048.0000000100', '00878-XB4GS9ShBRE.0000000005.0000001000', '00831-yr17PDCnDDW.0000000029.0000001000', '00844-q5QZSEeHe5g.0000000006.0000001000', '00813-svBbv1Pavdk.0000000021.0000000100', '00844-q5QZSEeHe5g.0000000013.0000000100', '00810-CrMo8WxCyVb.0000000013.0000001000', '00831-yr17PDCnDDW.0000000030.0000000100', '00873-bxsVRursffK.0000000003.0000000100', '00871-VBzV5z6i1WS.0000000047.0000000100', '00871-VBzV5z6i1WS.0000000000.0000001000', '00808-y9hTuugGdiq.0000000043.0000000100', '00835-q3zU7Yy5E5s.0000000030.0000000100', '00808-y9hTuugGdiq.0000000015.0000001000', '00815-h1zeeAwLh9Z.0000000044.0000001000', '00808-y9hTuugGdiq.0000000015.0000000100', '00843-DYehNKdT76V.0000000042.0000000100', '00813-svBbv1Pavdk.0000000040.0000000100', '00873-bxsVRursffK.0000000012.0000001000', '00839-zt1RVoi7PcG.0000000042.0000001000', '00871-VBzV5z6i1WS.0000000004.0000001000', '00813-svBbv1Pavdk.0000000028.0000001000', '00871-VBzV5z6i1WS.0000000000.0000000100', '00808-y9hTuugGdiq.0000000041.0000000100', '00813-svBbv1Pavdk.0000000021.0000001000', '00810-CrMo8WxCyVb.0000000000.0000001000', '00810-CrMo8WxCyVb.0000000034.0000001000', '00849-a8BtkwhxdRV.0000000030.0000001000', '00808-y9hTuugGdiq.0000000024.0000000100', '00844-q5QZSEeHe5g.0000000034.0000000100', '00810-CrMo8WxCyVb.0000000041.0000000100', '00835-q3zU7Yy5E5s.0000000030.0000001000', '00813-svBbv1Pavdk.0000000019.0000000100', '00813-svBbv1Pavdk.0000000003.0000001000', '00827-BAbdmeyTvMZ.0000000026.0000000100', '00848-ziup5kvtCCR.0000000006.0000001000', '00835-q3zU7Yy5E5s.0000000029.0000001000', '00810-CrMo8WxCyVb.0000000045.0000001000', '00871-VBzV5z6i1WS.0000000040.0000001000', '00810-CrMo8WxCyVb.0000000008.0000000100', '00813-svBbv1Pavdk.0000000019.0000001000', '00873-bxsVRursffK.0000000032.0000001000', '00873-bxsVRursffK.0000000006.0000001000', '00813-svBbv1Pavdk.0000000048.0000001000', '00831-yr17PDCnDDW.0000000024.0000000100', '00873-bxsVRursffK.0000000034.0000001000', '00849-a8BtkwhxdRV.0000000012.0000000100', '00813-svBbv1Pavdk.0000000047.0000001000', '00832-qyAac8rV8Zk.0000000045.0000001000', '00813-svBbv1Pavdk.0000000044.0000000100', '00878-XB4GS9ShBRE.0000000012.0000001000', '00878-XB4GS9ShBRE.0000000046.0000001000', '00843-DYehNKdT76V.0000000013.0000001000', '00839-zt1RVoi7PcG.0000000043.0000000100', '00844-q5QZSEeHe5g.0000000026.0000001000', '00815-h1zeeAwLh9Z.0000000024.0000000100', '00813-svBbv1Pavdk.0000000046.0000001000', '00873-bxsVRursffK.0000000024.0000000100', '00871-VBzV5z6i1WS.0000000036.0000000100', '00873-bxsVRursffK.0000000047.0000001000', '00871-VBzV5z6i1WS.0000000043.0000000100', '00843-DYehNKdT76V.0000000013.0000000100', '00827-BAbdmeyTvMZ.0000000046.0000000100', '00831-yr17PDCnDDW.0000000029.0000000100', '00843-DYehNKdT76V.0000000027.0000001000', '00831-yr17PDCnDDW.0000000013.0000001000', '00839-zt1RVoi7PcG.0000000043.0000001000', '00835-q3zU7Yy5E5s.0000000015.0000001000', '00848-ziup5kvtCCR.0000000019.0000001000', '00891-cvZr5TUy5C5.0000000005.0000001000', '00871-VBzV5z6i1WS.0000000034.0000001000', '00839-zt1RVoi7PcG.0000000010.0000001000', '00810-CrMo8WxCyVb.0000000022.0000000100', '00827-BAbdmeyTvMZ.0000000049.0000000100', '00815-h1zeeAwLh9Z.0000000027.0000000100', '00827-BAbdmeyTvMZ.0000000028.0000001000', '00813-svBbv1Pavdk.0000000013.0000000100', '00849-a8BtkwhxdRV.0000000005.0000001000', '00878-XB4GS9ShBRE.0000000009.0000001000', '00891-cvZr5TUy5C5.0000000044.0000001000', '00835-q3zU7Yy5E5s.0000000008.0000000100', '00835-q3zU7Yy5E5s.0000000043.0000001000', '00843-DYehNKdT76V.0000000023.0000000100', '00843-DYehNKdT76V.0000000029.0000000100', '00844-q5QZSEeHe5g.0000000010.0000000100', '00891-cvZr5TUy5C5.0000000049.0000001000', '00827-BAbdmeyTvMZ.0000000042.0000000100', '00832-qyAac8rV8Zk.0000000000.0000001000', '00849-a8BtkwhxdRV.0000000010.0000000100', '00848-ziup5kvtCCR.0000000004.0000000100', '00815-h1zeeAwLh9Z.0000000034.0000000100', '00839-zt1RVoi7PcG.0000000034.0000000100', '00891-cvZr5TUy5C5.0000000005.0000000100', '00849-a8BtkwhxdRV.0000000021.0000001000', '00810-CrMo8WxCyVb.0000000034.0000000100', '00848-ziup5kvtCCR.0000000037.0000000100', '00871-VBzV5z6i1WS.0000000015.0000000100', '00844-q5QZSEeHe5g.0000000016.0000000100', '00844-q5QZSEeHe5g.0000000036.0000001000', '00873-bxsVRursffK.0000000029.0000001000', '00827-BAbdmeyTvMZ.0000000007.0000001000', '00810-CrMo8WxCyVb.0000000028.0000001000', '00813-svBbv1Pavdk.0000000043.0000000100', '00810-CrMo8WxCyVb.0000000035.0000000100', '00835-q3zU7Yy5E5s.0000000007.0000001000', '00844-q5QZSEeHe5g.0000000015.0000000100', '00831-yr17PDCnDDW.0000000002.0000000100', '00815-h1zeeAwLh9Z.0000000024.0000001000', '00835-q3zU7Yy5E5s.0000000041.0000000100', '00871-VBzV5z6i1WS.0000000036.0000001000', '00808-y9hTuugGdiq.0000000049.0000000100', '00827-BAbdmeyTvMZ.0000000028.0000000100', '00839-zt1RVoi7PcG.0000000018.0000000100', '00810-CrMo8WxCyVb.0000000018.0000001000', '00832-qyAac8rV8Zk.0000000019.0000000100', '00871-VBzV5z6i1WS.0000000041.0000001000', '00835-q3zU7Yy5E5s.0000000049.0000000100', '00843-DYehNKdT76V.0000000039.0000001000', '00844-q5QZSEeHe5g.0000000028.0000001000', '00832-qyAac8rV8Zk.0000000005.0000001000', '00810-CrMo8WxCyVb.0000000028.0000000100', '00808-y9hTuugGdiq.0000000013.0000001000', '00810-CrMo8WxCyVb.0000000008.0000001000', '00839-zt1RVoi7PcG.0000000018.0000001000', '00815-h1zeeAwLh9Z.0000000017.0000000100', '00848-ziup5kvtCCR.0000000048.0000001000', '00848-ziup5kvtCCR.0000000008.0000001000', '00871-VBzV5z6i1WS.0000000021.0000001000', '00831-yr17PDCnDDW.0000000027.0000001000', '00849-a8BtkwhxdRV.0000000048.0000000100', '00843-DYehNKdT76V.0000000020.0000000100', '00813-svBbv1Pavdk.0000000004.0000001000', '00831-yr17PDCnDDW.0000000018.0000001000', '00873-bxsVRursffK.0000000015.0000000100', '00871-VBzV5z6i1WS.0000000010.0000000100', '00808-y9hTuugGdiq.0000000005.0000000100', '00844-q5QZSEeHe5g.0000000038.0000001000', '00871-VBzV5z6i1WS.0000000031.0000000100', '00810-CrMo8WxCyVb.0000000011.0000000100', '00839-zt1RVoi7PcG.0000000014.0000000100', '00849-a8BtkwhxdRV.0000000037.0000000100', '00871-VBzV5z6i1WS.0000000012.0000001000', '00823-7MXmsvcQjpJ.0000000044.0000001000', '00823-7MXmsvcQjpJ.0000000040.0000001000', '00835-q3zU7Yy5E5s.0000000042.0000001000', '00849-a8BtkwhxdRV.0000000015.0000000100', '00810-CrMo8WxCyVb.0000000029.0000000100', '00878-XB4GS9ShBRE.0000000029.0000001000', '00808-y9hTuugGdiq.0000000013.0000000100', '00891-cvZr5TUy5C5.0000000048.0000000100', '00827-BAbdmeyTvMZ.0000000023.0000000100', '00878-XB4GS9ShBRE.0000000048.0000001000', '00871-VBzV5z6i1WS.0000000038.0000001000', '00835-q3zU7Yy5E5s.0000000036.0000001000', '00848-ziup5kvtCCR.0000000004.0000001000', '00891-cvZr5TUy5C5.0000000047.0000001000', '00810-CrMo8WxCyVb.0000000025.0000000100', '00808-y9hTuugGdiq.0000000023.0000001000', '00835-q3zU7Yy5E5s.0000000028.0000001000', '00808-y9hTuugGdiq.0000000006.0000001000', '00815-h1zeeAwLh9Z.0000000043.0000000100', '00844-q5QZSEeHe5g.0000000047.0000000100', '00813-svBbv1Pavdk.0000000009.0000000100', '00827-BAbdmeyTvMZ.0000000045.0000000100', '00848-ziup5kvtCCR.0000000048.0000000100', '00873-bxsVRursffK.0000000035.0000000100', '00891-cvZr5TUy5C5.0000000025.0000001000', '00839-zt1RVoi7PcG.0000000033.0000000100', '00878-XB4GS9ShBRE.0000000009.0000000100', '00873-bxsVRursffK.0000000041.0000000100', '00849-a8BtkwhxdRV.0000000007.0000001000', '00871-VBzV5z6i1WS.0000000035.0000000100', '00815-h1zeeAwLh9Z.0000000044.0000000100', '00810-CrMo8WxCyVb.0000000027.0000000100', '00839-zt1RVoi7PcG.0000000042.0000000100', '00835-q3zU7Yy5E5s.0000000004.0000000100', '00844-q5QZSEeHe5g.0000000019.0000001000', '00835-q3zU7Yy5E5s.0000000046.0000001000', '00873-bxsVRursffK.0000000011.0000001000', '00849-a8BtkwhxdRV.0000000006.0000000100', '00844-q5QZSEeHe5g.0000000045.0000001000', '00835-q3zU7Yy5E5s.0000000039.0000000100', '00823-7MXmsvcQjpJ.0000000005.0000001000', '00873-bxsVRursffK.0000000041.0000001000', '00810-CrMo8WxCyVb.0000000048.0000000100', '00832-qyAac8rV8Zk.0000000010.0000000100', '00823-7MXmsvcQjpJ.0000000003.0000000100', '00810-CrMo8WxCyVb.0000000031.0000000100', '00839-zt1RVoi7PcG.0000000038.0000001000', '00891-cvZr5TUy5C5.0000000042.0000000100', '00827-BAbdmeyTvMZ.0000000045.0000001000', '00835-q3zU7Yy5E5s.0000000038.0000000100', '00835-q3zU7Yy5E5s.0000000003.0000000100', '00823-7MXmsvcQjpJ.0000000014.0000001000', '00835-q3zU7Yy5E5s.0000000000.0000001000', '00832-qyAac8rV8Zk.0000000048.0000001000', '00810-CrMo8WxCyVb.0000000046.0000000100', '00831-yr17PDCnDDW.0000000046.0000000100', '00843-DYehNKdT76V.0000000049.0000001000', '00823-7MXmsvcQjpJ.0000000044.0000000100', '00835-q3zU7Yy5E5s.0000000048.0000001000', '00891-cvZr5TUy5C5.0000000033.0000000100', '00815-h1zeeAwLh9Z.0000000036.0000001000', '00810-CrMo8WxCyVb.0000000047.0000001000', '00813-svBbv1Pavdk.0000000024.0000000100', '00810-CrMo8WxCyVb.0000000003.0000000100', '00844-q5QZSEeHe5g.0000000018.0000001000', '00848-ziup5kvtCCR.0000000038.0000001000', '00848-ziup5kvtCCR.0000000031.0000000100', '00808-y9hTuugGdiq.0000000007.0000000100', '00891-cvZr5TUy5C5.0000000010.0000000100', '00835-q3zU7Yy5E5s.0000000021.0000001000', '00873-bxsVRursffK.0000000013.0000001000', '00848-ziup5kvtCCR.0000000024.0000001000', '00823-7MXmsvcQjpJ.0000000033.0000001000', '00871-VBzV5z6i1WS.0000000032.0000001000', '00849-a8BtkwhxdRV.0000000033.0000000100', '00873-bxsVRursffK.0000000039.0000000100', '00839-zt1RVoi7PcG.0000000015.0000001000', '00873-bxsVRursffK.0000000047.0000000100', '00832-qyAac8rV8Zk.0000000020.0000000100', '00843-DYehNKdT76V.0000000023.0000001000', '00808-y9hTuugGdiq.0000000008.0000000100', '00827-BAbdmeyTvMZ.0000000029.0000000100', '00871-VBzV5z6i1WS.0000000012.0000000100', '00849-a8BtkwhxdRV.0000000045.0000001000', '00827-BAbdmeyTvMZ.0000000021.0000000100', '00813-svBbv1Pavdk.0000000016.0000000100', '00873-bxsVRursffK.0000000001.0000001000', '00823-7MXmsvcQjpJ.0000000013.0000000100', '00827-BAbdmeyTvMZ.0000000025.0000001000', '00843-DYehNKdT76V.0000000014.0000000100', '00871-VBzV5z6i1WS.0000000031.0000001000', '00878-XB4GS9ShBRE.0000000013.0000000100', '00891-cvZr5TUy5C5.0000000019.0000000100', '00815-h1zeeAwLh9Z.0000000018.0000000100', '00844-q5QZSEeHe5g.0000000026.0000000100', '00891-cvZr5TUy5C5.0000000015.0000001000', '00813-svBbv1Pavdk.0000000042.0000000100', '00815-h1zeeAwLh9Z.0000000037.0000000100', '00810-CrMo8WxCyVb.0000000031.0000001000', '00843-DYehNKdT76V.0000000001.0000001000', '00848-ziup5kvtCCR.0000000022.0000000100', '00873-bxsVRursffK.0000000005.0000001000', '00808-y9hTuugGdiq.0000000004.0000000100', '00844-q5QZSEeHe5g.0000000023.0000000100', '00844-q5QZSEeHe5g.0000000037.0000000100', '00878-XB4GS9ShBRE.0000000037.0000000100', '00873-bxsVRursffK.0000000029.0000000100', '00848-ziup5kvtCCR.0000000040.0000000100', '00843-DYehNKdT76V.0000000034.0000000100', '00844-q5QZSEeHe5g.0000000013.0000001000', '00873-bxsVRursffK.0000000028.0000001000', '00891-cvZr5TUy5C5.0000000042.0000001000', '00873-bxsVRursffK.0000000008.0000001000', '00871-VBzV5z6i1WS.0000000015.0000001000', '00891-cvZr5TUy5C5.0000000048.0000001000', '00813-svBbv1Pavdk.0000000044.0000001000', '00832-qyAac8rV8Zk.0000000049.0000001000', '00848-ziup5kvtCCR.0000000040.0000001000', '00871-VBzV5z6i1WS.0000000009.0000001000', '00813-svBbv1Pavdk.0000000039.0000001000', '00891-cvZr5TUy5C5.0000000019.0000001000', '00813-svBbv1Pavdk.0000000030.0000001000', '00835-q3zU7Yy5E5s.0000000039.0000001000', '00843-DYehNKdT76V.0000000017.0000001000', '00827-BAbdmeyTvMZ.0000000031.0000000100', '00873-bxsVRursffK.0000000019.0000001000', '00891-cvZr5TUy5C5.0000000041.0000001000', '00832-qyAac8rV8Zk.0000000036.0000000100', '00808-y9hTuugGdiq.0000000018.0000000100', '00878-XB4GS9ShBRE.0000000015.0000000100', '00832-qyAac8rV8Zk.0000000034.0000000100', '00835-q3zU7Yy5E5s.0000000028.0000000100', '00873-bxsVRursffK.0000000001.0000000100', '00849-a8BtkwhxdRV.0000000005.0000000100', '00808-y9hTuugGdiq.0000000041.0000001000', '00813-svBbv1Pavdk.0000000043.0000001000', '00835-q3zU7Yy5E5s.0000000046.0000000100', '00831-yr17PDCnDDW.0000000024.0000001000', '00827-BAbdmeyTvMZ.0000000042.0000001000', '00835-q3zU7Yy5E5s.0000000049.0000001000', '00871-VBzV5z6i1WS.0000000027.0000001000', '00823-7MXmsvcQjpJ.0000000009.0000000100', '00823-7MXmsvcQjpJ.0000000033.0000000100', '00871-VBzV5z6i1WS.0000000034.0000000100', '00813-svBbv1Pavdk.0000000020.0000001000', '00871-VBzV5z6i1WS.0000000033.0000001000', '00835-q3zU7Yy5E5s.0000000020.0000001000', '00843-DYehNKdT76V.0000000045.0000001000', '00835-q3zU7Yy5E5s.0000000008.0000001000', '00878-XB4GS9ShBRE.0000000006.0000001000', '00843-DYehNKdT76V.0000000026.0000000100', '00835-q3zU7Yy5E5s.0000000036.0000000100', '00839-zt1RVoi7PcG.0000000022.0000001000', '00832-qyAac8rV8Zk.0000000027.0000000100', '00848-ziup5kvtCCR.0000000045.0000000100', '00815-h1zeeAwLh9Z.0000000042.0000000100', '00873-bxsVRursffK.0000000006.0000000100', '00849-a8BtkwhxdRV.0000000006.0000001000', '00843-DYehNKdT76V.0000000036.0000001000', '00871-VBzV5z6i1WS.0000000005.0000001000', '00839-zt1RVoi7PcG.0000000046.0000000100', '00827-BAbdmeyTvMZ.0000000025.0000000100', '00891-cvZr5TUy5C5.0000000036.0000000100', '00808-y9hTuugGdiq.0000000022.0000001000', '00843-DYehNKdT76V.0000000001.0000000100', '00871-VBzV5z6i1WS.0000000002.0000000100', '00844-q5QZSEeHe5g.0000000036.0000000100', '00891-cvZr5TUy5C5.0000000036.0000001000', '00835-q3zU7Yy5E5s.0000000006.0000000100', '00823-7MXmsvcQjpJ.0000000009.0000001000', '00839-zt1RVoi7PcG.0000000027.0000001000', '00848-ziup5kvtCCR.0000000011.0000001000', '00823-7MXmsvcQjpJ.0000000048.0000001000', '00878-XB4GS9ShBRE.0000000031.0000001000', '00808-y9hTuugGdiq.0000000022.0000000100', '00844-q5QZSEeHe5g.0000000014.0000000100', '00843-DYehNKdT76V.0000000036.0000000100', '00835-q3zU7Yy5E5s.0000000041.0000001000', '00873-bxsVRursffK.0000000035.0000001000', '00832-qyAac8rV8Zk.0000000010.0000001000', '00848-ziup5kvtCCR.0000000038.0000000100', '00878-XB4GS9ShBRE.0000000013.0000001000', '00823-7MXmsvcQjpJ.0000000030.0000001000', '00808-y9hTuugGdiq.0000000011.0000000100', '00808-y9hTuugGdiq.0000000045.0000000100', '00813-svBbv1Pavdk.0000000039.0000000100', '00832-qyAac8rV8Zk.0000000024.0000001000', '00827-BAbdmeyTvMZ.0000000007.0000000100', '00848-ziup5kvtCCR.0000000001.0000000100', '00891-cvZr5TUy5C5.0000000049.0000000100', '00849-a8BtkwhxdRV.0000000034.0000000100', '00871-VBzV5z6i1WS.0000000048.0000001000', '00839-zt1RVoi7PcG.0000000027.0000000100', '00808-y9hTuugGdiq.0000000024.0000001000', '00835-q3zU7Yy5E5s.0000000005.0000000100', '00878-XB4GS9ShBRE.0000000006.0000000100', '00871-VBzV5z6i1WS.0000000027.0000000100', '00873-bxsVRursffK.0000000014.0000000100', '00844-q5QZSEeHe5g.0000000020.0000001000', '00823-7MXmsvcQjpJ.0000000013.0000001000', '00835-q3zU7Yy5E5s.0000000043.0000000100', '00832-qyAac8rV8Zk.0000000019.0000001000', '00843-DYehNKdT76V.0000000045.0000000100', '00843-DYehNKdT76V.0000000014.0000001000', '00843-DYehNKdT76V.0000000016.0000000100', '00839-zt1RVoi7PcG.0000000031.0000001000', '00891-cvZr5TUy5C5.0000000025.0000000100', '00891-cvZr5TUy5C5.0000000023.0000001000', '00849-a8BtkwhxdRV.0000000045.0000000100', '00848-ziup5kvtCCR.0000000015.0000000100', '00810-CrMo8WxCyVb.0000000022.0000001000', '00878-XB4GS9ShBRE.0000000046.0000000100', '00849-a8BtkwhxdRV.0000000027.0000001000', '00808-y9hTuugGdiq.0000000048.0000001000', '00808-y9hTuugGdiq.0000000017.0000001000', '00843-DYehNKdT76V.0000000031.0000001000', '00832-qyAac8rV8Zk.0000000040.0000000100', '00835-q3zU7Yy5E5s.0000000027.0000001000', '00815-h1zeeAwLh9Z.0000000036.0000000100', '00848-ziup5kvtCCR.0000000024.0000000100', '00808-y9hTuugGdiq.0000000031.0000001000', '00871-VBzV5z6i1WS.0000000042.0000001000', '00839-zt1RVoi7PcG.0000000009.0000000100', '00835-q3zU7Yy5E5s.0000000042.0000000100', '00832-qyAac8rV8Zk.0000000020.0000001000', '00849-a8BtkwhxdRV.0000000015.0000001000', '00808-y9hTuugGdiq.0000000018.0000001000', '00878-XB4GS9ShBRE.0000000021.0000001000', '00873-bxsVRursffK.0000000013.0000000100', '00878-XB4GS9ShBRE.0000000015.0000001000', '00810-CrMo8WxCyVb.0000000039.0000001000', '00839-zt1RVoi7PcG.0000000010.0000000100', '00835-q3zU7Yy5E5s.0000000020.0000000100', '00831-yr17PDCnDDW.0000000027.0000000100', '00823-7MXmsvcQjpJ.0000000035.0000000100', '00832-qyAac8rV8Zk.0000000032.0000001000', '00878-XB4GS9ShBRE.0000000012.0000000100', '00810-CrMo8WxCyVb.0000000011.0000001000', '00823-7MXmsvcQjpJ.0000000007.0000001000', '00813-svBbv1Pavdk.0000000040.0000001000', '00844-q5QZSEeHe5g.0000000034.0000001000', '00808-y9hTuugGdiq.0000000046.0000001000', '00873-bxsVRursffK.0000000008.0000000100', '00878-XB4GS9ShBRE.0000000028.0000001000', '00808-y9hTuugGdiq.0000000046.0000000100', '00849-a8BtkwhxdRV.0000000024.0000000100', '00810-CrMo8WxCyVb.0000000047.0000000100', '00849-a8BtkwhxdRV.0000000027.0000000100', '00832-qyAac8rV8Zk.0000000034.0000001000', '00810-CrMo8WxCyVb.0000000041.0000001000', '00832-qyAac8rV8Zk.0000000048.0000000100', '00849-a8BtkwhxdRV.0000000035.0000001000', '00813-svBbv1Pavdk.0000000041.0000000100', '00871-VBzV5z6i1WS.0000000043.0000001000', '00871-VBzV5z6i1WS.0000000010.0000001000', '00832-qyAac8rV8Zk.0000000022.0000000100', '00848-ziup5kvtCCR.0000000019.0000000100', '00831-yr17PDCnDDW.0000000018.0000000100', '00871-VBzV5z6i1WS.0000000023.0000001000', '00810-CrMo8WxCyVb.0000000029.0000001000', '00844-q5QZSEeHe5g.0000000045.0000000100', '00808-y9hTuugGdiq.0000000048.0000000100', '00873-bxsVRursffK.0000000023.0000001000', '00839-zt1RVoi7PcG.0000000000.0000001000', '00878-XB4GS9ShBRE.0000000036.0000000100', '00839-zt1RVoi7PcG.0000000014.0000001000', '00873-bxsVRursffK.0000000028.0000000100', '00848-ziup5kvtCCR.0000000030.0000000100', '00808-y9hTuugGdiq.0000000025.0000001000', '00849-a8BtkwhxdRV.0000000048.0000001000', '00839-zt1RVoi7PcG.0000000031.0000000100', '00878-XB4GS9ShBRE.0000000005.0000000100', '00878-XB4GS9ShBRE.0000000023.0000000100', '00832-qyAac8rV8Zk.0000000003.0000001000', '00815-h1zeeAwLh9Z.0000000032.0000000100', '00827-BAbdmeyTvMZ.0000000026.0000001000', '00810-CrMo8WxCyVb.0000000013.0000000100', '00839-zt1RVoi7PcG.0000000028.0000000100', '00848-ziup5kvtCCR.0000000022.0000001000', '00848-ziup5kvtCCR.0000000003.0000000100', '00808-y9hTuugGdiq.0000000008.0000001000', '00835-q3zU7Yy5E5s.0000000047.0000000100', '00832-qyAac8rV8Zk.0000000040.0000001000', '00844-q5QZSEeHe5g.0000000002.0000000100', '00848-ziup5kvtCCR.0000000030.0000001000', '00823-7MXmsvcQjpJ.0000000020.0000001000', '00827-BAbdmeyTvMZ.0000000017.0000000100', '00849-a8BtkwhxdRV.0000000013.0000000100', '00815-h1zeeAwLh9Z.0000000033.0000000100', '00831-yr17PDCnDDW.0000000002.0000001000', '00891-cvZr5TUy5C5.0000000026.0000001000', '00835-q3zU7Yy5E5s.0000000047.0000001000', '00891-cvZr5TUy5C5.0000000021.0000001000', '00808-y9hTuugGdiq.0000000036.0000000100', '00844-q5QZSEeHe5g.0000000018.0000000100', '00844-q5QZSEeHe5g.0000000048.0000001000', '00873-bxsVRursffK.0000000048.0000001000', '00878-XB4GS9ShBRE.0000000021.0000000100', '00873-bxsVRursffK.0000000036.0000001000', '00835-q3zU7Yy5E5s.0000000005.0000001000', '00831-yr17PDCnDDW.0000000031.0000001000', '00871-VBzV5z6i1WS.0000000021.0000000100', '00813-svBbv1Pavdk.0000000046.0000000100', '00871-VBzV5z6i1WS.0000000030.0000001000', '00844-q5QZSEeHe5g.0000000006.0000000100', '00844-q5QZSEeHe5g.0000000020.0000000100', '00891-cvZr5TUy5C5.0000000010.0000001000', '00843-DYehNKdT76V.0000000042.0000001000', '00871-VBzV5z6i1WS.0000000018.0000001000', '00815-h1zeeAwLh9Z.0000000039.0000000100', '00813-svBbv1Pavdk.0000000001.0000000100', '00878-XB4GS9ShBRE.0000000037.0000001000', '00815-h1zeeAwLh9Z.0000000034.0000001000', '00839-zt1RVoi7PcG.0000000033.0000001000', '00835-q3zU7Yy5E5s.0000000016.0000000100', '00871-VBzV5z6i1WS.0000000018.0000000100', '00832-qyAac8rV8Zk.0000000002.0000000100', '00891-cvZr5TUy5C5.0000000041.0000000100', '00835-q3zU7Yy5E5s.0000000032.0000001000', '00835-q3zU7Yy5E5s.0000000026.0000001000', '00839-zt1RVoi7PcG.0000000041.0000000100', '00873-bxsVRursffK.0000000039.0000001000', '00844-q5QZSEeHe5g.0000000038.0000000100', '00835-q3zU7Yy5E5s.0000000016.0000001000', '00891-cvZr5TUy5C5.0000000044.0000000100', '00827-BAbdmeyTvMZ.0000000017.0000001000', '00844-q5QZSEeHe5g.0000000014.0000001000', '00873-bxsVRursffK.0000000032.0000000100', '00808-y9hTuugGdiq.0000000031.0000000100', '00835-q3zU7Yy5E5s.0000000000.0000000100', '00891-cvZr5TUy5C5.0000000002.0000001000', '00831-yr17PDCnDDW.0000000004.0000000100', '00878-XB4GS9ShBRE.0000000036.0000001000', '00878-XB4GS9ShBRE.0000000020.0000001000', '00808-y9hTuugGdiq.0000000017.0000000100', '00815-h1zeeAwLh9Z.0000000028.0000001000', '00891-cvZr5TUy5C5.0000000015.0000000100', '00839-zt1RVoi7PcG.0000000038.0000000100', '00835-q3zU7Yy5E5s.0000000017.0000000100', '00831-yr17PDCnDDW.0000000030.0000001000', '00835-q3zU7Yy5E5s.0000000010.0000001000', '00878-XB4GS9ShBRE.0000000007.0000000100', '00849-a8BtkwhxdRV.0000000030.0000000100', '00810-CrMo8WxCyVb.0000000048.0000001000', '00878-XB4GS9ShBRE.0000000003.0000001000', '00835-q3zU7Yy5E5s.0000000027.0000000100', '00843-DYehNKdT76V.0000000020.0000001000', '00843-DYehNKdT76V.0000000030.0000000100', '00835-q3zU7Yy5E5s.0000000033.0000000100', '00871-VBzV5z6i1WS.0000000035.0000001000', '00839-zt1RVoi7PcG.0000000034.0000001000', '00835-q3zU7Yy5E5s.0000000010.0000000100', '00815-h1zeeAwLh9Z.0000000039.0000001000', '00831-yr17PDCnDDW.0000000046.0000001000', '00873-bxsVRursffK.0000000015.0000001000', '00813-svBbv1Pavdk.0000000042.0000001000', '00831-yr17PDCnDDW.0000000048.0000001000', '00843-DYehNKdT76V.0000000009.0000001000', '00815-h1zeeAwLh9Z.0000000037.0000001000', '00878-XB4GS9ShBRE.0000000018.0000001000', '00815-h1zeeAwLh9Z.0000000018.0000001000', '00843-DYehNKdT76V.0000000011.0000000100', '00808-y9hTuugGdiq.0000000025.0000000100', '00823-7MXmsvcQjpJ.0000000005.0000000100', '00871-VBzV5z6i1WS.0000000038.0000000100', '00813-svBbv1Pavdk.0000000009.0000001000', '00848-ziup5kvtCCR.0000000012.0000000100', '00844-q5QZSEeHe5g.0000000044.0000000100', '00813-svBbv1Pavdk.0000000007.0000000100', '00835-q3zU7Yy5E5s.0000000024.0000000100', '00843-DYehNKdT76V.0000000044.0000001000', '00835-q3zU7Yy5E5s.0000000004.0000001000', '00844-q5QZSEeHe5g.0000000019.0000000100', '00832-qyAac8rV8Zk.0000000049.0000000100', '00849-a8BtkwhxdRV.0000000021.0000000100', '00823-7MXmsvcQjpJ.0000000026.0000001000', '00823-7MXmsvcQjpJ.0000000040.0000000100', '00835-q3zU7Yy5E5s.0000000026.0000000100', '00813-svBbv1Pavdk.0000000016.0000001000', '00810-CrMo8WxCyVb.0000000045.0000000100', '00823-7MXmsvcQjpJ.0000000035.0000001000', '00873-bxsVRursffK.0000000027.0000001000', '00835-q3zU7Yy5E5s.0000000007.0000000100', '00808-y9hTuugGdiq.0000000045.0000001000', '00844-q5QZSEeHe5g.0000000002.0000001000', '00848-ziup5kvtCCR.0000000012.0000001000', '00873-bxsVRursffK.0000000011.0000000100', '00823-7MXmsvcQjpJ.0000000043.0000000100', '00871-VBzV5z6i1WS.0000000029.0000001000', '00808-y9hTuugGdiq.0000000043.0000001000', '00873-bxsVRursffK.0000000012.0000000100', '00815-h1zeeAwLh9Z.0000000014.0000001000', '00815-h1zeeAwLh9Z.0000000035.0000000100', '00848-ziup5kvtCCR.0000000037.0000001000', '00843-DYehNKdT76V.0000000026.0000001000', '00813-svBbv1Pavdk.0000000041.0000001000', '00808-y9hTuugGdiq.0000000023.0000000100', '00813-svBbv1Pavdk.0000000048.0000000100', '00827-BAbdmeyTvMZ.0000000034.0000000100', '00832-qyAac8rV8Zk.0000000045.0000000100', '00849-a8BtkwhxdRV.0000000023.0000000100', '00810-CrMo8WxCyVb.0000000018.0000000100', '00813-svBbv1Pavdk.0000000013.0000001000', '00815-h1zeeAwLh9Z.0000000017.0000001000', '00810-CrMo8WxCyVb.0000000025.0000001000', '00823-7MXmsvcQjpJ.0000000020.0000000100', '00839-zt1RVoi7PcG.0000000025.0000001000', '00871-VBzV5z6i1WS.0000000030.0000000100', '00827-BAbdmeyTvMZ.0000000002.0000000100', '00843-DYehNKdT76V.0000000005.0000001000', '00815-h1zeeAwLh9Z.0000000021.0000001000', '00827-BAbdmeyTvMZ.0000000016.0000001000', '00832-qyAac8rV8Zk.0000000008.0000000100', '00815-h1zeeAwLh9Z.0000000002.0000001000', '00813-svBbv1Pavdk.0000000028.0000000100', '00849-a8BtkwhxdRV.0000000034.0000001000', '00839-zt1RVoi7PcG.0000000041.0000001000', '00835-q3zU7Yy5E5s.0000000017.0000001000', '00823-7MXmsvcQjpJ.0000000007.0000000100', '00843-DYehNKdT76V.0000000022.0000000100', '00832-qyAac8rV8Zk.0000000022.0000001000', '00815-h1zeeAwLh9Z.0000000028.0000000100', '00810-CrMo8WxCyVb.0000000001.0000001000', '00810-CrMo8WxCyVb.0000000035.0000001000', '00832-qyAac8rV8Zk.0000000030.0000001000', '00871-VBzV5z6i1WS.0000000042.0000000100', '00815-h1zeeAwLh9Z.0000000042.0000001000', '00813-svBbv1Pavdk.0000000024.0000001000', '00891-cvZr5TUy5C5.0000000002.0000000100', '00832-qyAac8rV8Zk.0000000021.0000001000', '00839-zt1RVoi7PcG.0000000028.0000001000', '00848-ziup5kvtCCR.0000000000.0000001000', '00835-q3zU7Yy5E5s.0000000032.0000000100', '00839-zt1RVoi7PcG.0000000022.0000000100', '00832-qyAac8rV8Zk.0000000007.0000001000', '00844-q5QZSEeHe5g.0000000047.0000001000', '00832-qyAac8rV8Zk.0000000005.0000000100', '00813-svBbv1Pavdk.0000000020.0000000100', '00891-cvZr5TUy5C5.0000000040.0000000100', '00873-bxsVRursffK.0000000003.0000001000', '00844-q5QZSEeHe5g.0000000037.0000001000', '00849-a8BtkwhxdRV.0000000023.0000001000', '00891-cvZr5TUy5C5.0000000047.0000000100', '00849-a8BtkwhxdRV.0000000024.0000001000', '00827-BAbdmeyTvMZ.0000000016.0000000100', '00844-q5QZSEeHe5g.0000000010.0000001000', '00873-bxsVRursffK.0000000034.0000000100', '00823-7MXmsvcQjpJ.0000000002.0000001000', '00848-ziup5kvtCCR.0000000031.0000001000', '00835-q3zU7Yy5E5s.0000000048.0000000100', '00815-h1zeeAwLh9Z.0000000021.0000000100', '00891-cvZr5TUy5C5.0000000028.0000001000', '00891-cvZr5TUy5C5.0000000023.0000000100', '00832-qyAac8rV8Zk.0000000021.0000000100', '00815-h1zeeAwLh9Z.0000000032.0000001000', '00878-XB4GS9ShBRE.0000000008.0000000100', '00815-h1zeeAwLh9Z.0000000035.0000001000', '00831-yr17PDCnDDW.0000000031.0000000100', '00823-7MXmsvcQjpJ.0000000018.0000001000', '00873-bxsVRursffK.0000000048.0000000100', '00815-h1zeeAwLh9Z.0000000027.0000001000', '00878-XB4GS9ShBRE.0000000007.0000001000', '00878-XB4GS9ShBRE.0000000018.0000000100', '00823-7MXmsvcQjpJ.0000000043.0000001000', '00843-DYehNKdT76V.0000000005.0000000100', '00823-7MXmsvcQjpJ.0000000003.0000001000', '00871-VBzV5z6i1WS.0000000009.0000000100', '00873-bxsVRursffK.0000000023.0000000100', '00839-zt1RVoi7PcG.0000000015.0000000100', '00808-y9hTuugGdiq.0000000004.0000001000', '00832-qyAac8rV8Zk.0000000003.0000000100', '00827-BAbdmeyTvMZ.0000000049.0000001000', '00843-DYehNKdT76V.0000000029.0000001000', '00873-bxsVRursffK.0000000027.0000000100', '00835-q3zU7Yy5E5s.0000000021.0000000100', '00835-q3zU7Yy5E5s.0000000038.0000001000', '00871-VBzV5z6i1WS.0000000017.0000000100', '00808-y9hTuugGdiq.0000000030.0000001000', '00815-h1zeeAwLh9Z.0000000030.0000000100', '00871-VBzV5z6i1WS.0000000033.0000000100', '00827-BAbdmeyTvMZ.0000000031.0000001000', '00843-DYehNKdT76V.0000000017.0000000100', '00813-svBbv1Pavdk.0000000001.0000001000', '00878-XB4GS9ShBRE.0000000023.0000001000', '00831-yr17PDCnDDW.0000000034.0000000100', '00839-zt1RVoi7PcG.0000000046.0000001000', '00871-VBzV5z6i1WS.0000000004.0000000100', '00844-q5QZSEeHe5g.0000000016.0000001000', '00878-XB4GS9ShBRE.0000000048.0000000100', '00871-VBzV5z6i1WS.0000000045.0000001000', '00849-a8BtkwhxdRV.0000000029.0000001000', '00808-y9hTuugGdiq.0000000006.0000000100', '00813-svBbv1Pavdk.0000000030.0000000100', '00831-yr17PDCnDDW.0000000037.0000000100', '00871-VBzV5z6i1WS.0000000047.0000001000', '00810-CrMo8WxCyVb.0000000027.0000001000', '00813-svBbv1Pavdk.0000000047.0000000100', '00823-7MXmsvcQjpJ.0000000002.0000000100', '00848-ziup5kvtCCR.0000000021.0000001000', '00815-h1zeeAwLh9Z.0000000002.0000000100', '00871-VBzV5z6i1WS.0000000017.0000001000', '00808-y9hTuugGdiq.0000000005.0000001000', '00848-ziup5kvtCCR.0000000021.0000000100', '00848-ziup5kvtCCR.0000000015.0000001000', '00835-q3zU7Yy5E5s.0000000024.0000001000', '00843-DYehNKdT76V.0000000044.0000000100', '00844-q5QZSEeHe5g.0000000028.0000000100', '00810-CrMo8WxCyVb.0000000039.0000000100', '00810-CrMo8WxCyVb.0000000000.0000000100', '00871-VBzV5z6i1WS.0000000005.0000000100', '00823-7MXmsvcQjpJ.0000000014.0000000100', '00832-qyAac8rV8Zk.0000000028.0000001000', '00832-qyAac8rV8Zk.0000000002.0000001000', '00832-qyAac8rV8Zk.0000000032.0000000100', '00844-q5QZSEeHe5g.0000000000.0000000100', '00849-a8BtkwhxdRV.0000000029.0000000100', '00871-VBzV5z6i1WS.0000000023.0000000100', '00849-a8BtkwhxdRV.0000000013.0000001000', '00831-yr17PDCnDDW.0000000037.0000001000', '00835-q3zU7Yy5E5s.0000000003.0000001000', '00878-XB4GS9ShBRE.0000000039.0000000100', '00871-VBzV5z6i1WS.0000000041.0000000100', '00844-q5QZSEeHe5g.0000000044.0000001000', '00832-qyAac8rV8Zk.0000000028.0000000100', '00873-bxsVRursffK.0000000014.0000001000']:
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
                # print(zs, video_name, k, sum_iou/denom if denom>0 else 0)

        
        for k in k_list:
            if (video_data["zero-shot"][k]["TP"] + video_data["zero-shot"][k]["FN"] \
                + video_data["non-zero-shot"][k]["TP"] + video_data["non-zero-shot"][k]["FN"]) > 0:
                sum_iou = video_data["zero-shot"][k]["IOU"] + video_data["non-zero-shot"][k]["IOU"]
                tp = video_data["zero-shot"][k]["TP"] + video_data["non-zero-shot"][k]["TP"]
                fn = video_data["zero-shot"][k]["FN"] + video_data["non-zero-shot"][k]["FN"]
                fp = video_data["zero-shot"][k]["FP"]
                assert video_data["zero-shot"][k]["FP"]==video_data["non-zero-shot"][k]["FP"]
                denom = tp + (0.5*fp) + (0.5*fn)
            else:
                sum_iou = 0
                denom = 0
            total["aggregate"][k]["sum_iou"] += sum_iou
            total["aggregate"][k]["sum_denom"] += denom

        v_tot+=1
        # print()
        # if v_tot>=87:
        #     break

    print(v_tot)
    for zs in ["zero-shot", "non-zero-shot", "aggregate"]:
        avg=0
        for k in k_list:
            ivpq_k = total[zs][k]["sum_iou"]/total[zs][k]["sum_denom"] if total[zs][k]["sum_denom"]>0 else 0
            avg+=ivpq_k
            print(zs, "Total", k, ivpq_k)
        print(zs,"Total", "iVPQ", avg/len(k_list))





if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rle_path',type=str, required=True) 
    parser.add_argument('--ref_path',type=str, required=True)
    parser.add_argument('--ref_split',type=str, required=True)
    parser.add_argument('--compute', action='store_true', default=False)
    parser.add_argument('--vkn', action='store_true', default=False)
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

    dataset = MVPDataset(root=args.ref_path,
                    split=args.ref_split,
                    use_stuff=False,
                    window_size = 0)


    if args.compute:
        if not args.vkn:
            evaluate_iVPQ_v1(in_rle_dir, out_json_dir, dataset, device, 0, 0)
        else:
            evaluate_iVPQ_vkn(in_rle_dir, out_json_dir, dataset, device, 0, 0)
    if args.summarize:
        summarize_iVPQ(out_json_dir, fpath="metrics_iVPQ_v1.json")
