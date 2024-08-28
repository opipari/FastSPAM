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




def collect_rle_tracks(video, video_rle_dir, threshold=0.5):
    track_inds = set()

    for f_idx in range(len(video)):
        rle_file = os.path.join(video_rle_dir, f'{f_idx:010d}.pt')
        track_boxes = torch.load(rle_file, map_location='cpu')['bbox_results']
        track_inds.update(np.flatnonzero(track_boxes[:,5]>threshold))

    return list(track_inds)


def collect_rle_window(video_rle_dir, default_size=(1,480,640)):
    rle_segments = []
    for name in sorted(os.listdir(video_rle_dir)):
        rle_file = os.path.join(video_rle_dir, name)
        rle_seg = metric_utils.read_panomaskRLE(rle_file)
        if len(rle_seg)==0:
            rle_seg = torch.zeros(default_size)
        rle_segments.append(rle_seg.to(torch.bool))
    return rle_segments


def get_tubes(rle_segments, tube_device='cpu', frame_device='cuda'):
    orig_size = rle_segments[0].shape[1:]

    tubes = {}
    prev_segments = []
    prev_segments_map = {}
    
    
    for frame_i in range(len(rle_segments)):
        curr_segments = rle_segments[frame_i]
        curr_segments_map = {}

        N = len(tubes.keys())

        if len(curr_segments)==0:
            for t_id in tubes:
                tubes[t_id].append(-1)

        elif len(prev_segments)==0:
            curr_segments = curr_segments.to(dtype=torch.bool).to(device=tube_device) # n x H x W
            curr_segments_map = {curr_ind: N+curr_ind for curr_ind in range(len(curr_segments))}

            for curr_ind in range(len(curr_segments)):
                t_id = N+curr_ind
                assert t_id not in tubes
                tubes[t_id] = [-1]*frame_i + [curr_ind]
        else:
            curr_segments = curr_segments.to(dtype=torch.bool).to(device=tube_device)

            # Match rle segments to reference segments, then merge unmatched rle segments
            (matched_prev_ind, unmatched_prev_ind), (matched_curr_ind, unmatched_curr_ind), _ = metric_utils.match_segments(prev_segments.to(device=frame_device), curr_segments.to(device=frame_device))

            update_t_ids = []
            for prev_ind, curr_ind in zip(matched_prev_ind, matched_curr_ind):
                t_id = prev_segments_map[prev_ind]
                assert t_id in tubes
                tubes[t_id].append(curr_ind)
                update_t_ids.append(t_id)
                curr_segments_map[curr_ind] = t_id

            for ind_, curr_ind in enumerate(unmatched_curr_ind):
                t_id = N+ind_
                assert t_id not in tubes
                tubes[t_id] = [-1]*frame_i + [curr_ind]
                update_t_ids.append(t_id)
                curr_segments_map[curr_ind] = t_id

            for t_id in tubes:
                if t_id in update_t_ids:
                    continue
                tubes[t_id].append(-1)

        prev_segments = curr_segments
        prev_segments_map = curr_segments_map

    return tubes


def evaluate_STQ(in_rle_dir, out_dir, dataset, conf_threshold=0.5, device='cpu'):
    
    rand_inds = [1410,  185, 1366,   52, 1736, 1672,  890,  455,  797,  453, 1664, 1382,
         910, 1332,  662,  520,  181,  702,  221,  807,  328, 1619,  750,  149,
         373,   60, 1470, 1018, 1593,  261, 1667, 1637, 1211, 1006,  780,  474,
         758,   47, 1462,  648,  108, 1383, 1258,  915,  481,  608, 1723, 1467,
        1118,  883, 1140, 1452,  874, 1191,  962, 1252,  169,  776,  920,  880,
         772,  803,  145,   69,  425,  922,  781,  548,  867, 1548, 1194,  727,
         712, 1532,  381, 1578,   94,   23, 1393,   61,  975, 1464, 1740, 1079,
        1613, 1500, 1098,  570, 1122,  500,  567,  615, 1046, 1275,  459, 1291,
         284, 1588,  102,  152,  856, 1400, 1663, 1210, 1640,  490,  970,  928,
        1376, 1580, 1065,  713,   11, 1068,  495,    9, 1012,  541,  724,    3,
         443,  349,  945, 1739,  613,  382,  320,  516,  301, 1606,  521, 1221,
        1090, 1013,  588,  265,  896,  610, 1715,  544,  321, 1517, 1523,  113,
         602, 1551,  538, 1161,  871,  642, 1409,  432, 1197, 1249,  987, 1322,
        1379, 1404, 1119,  283, 1016,  461,  263,  370,  894,  434,  545,  919,
        1255, 1328,  857, 1614,  484,  259]
    # for vii,video in enumerate(tqdm(dataset, position=0)):
    for vii, video_i in enumerate(tqdm(rand_inds, position=0)):
        video = dataset[video_i]
        sample = next(iter(video))
        video_name = sample['meta']['video_name']
        video_rle_dir = os.path.join(in_rle_dir, video_name)
        video_out_dir = os.path.join(out_dir, video_name)

        video_statistics = {"AQ": 0.0, "SQ_overlap": 0.0, "SQ_non_overlap":0.0, "N_tracks": 0.0, "track_stats": {}}
        track_aq_stats = video_statistics["track_stats"]

        
        sample_rle_file = os.path.join(video_rle_dir, sample['meta']['window_names'][0].split('.')[0]+'.pt')
        sample_rle = torch.load(sample_rle_file,map_location='cpu')
        if 'bbox_results' in sample_rle:
            track_inds = collect_rle_tracks(video, video_rle_dir)
            tubes = {k: [k]*len(video) for k in range(len(track_inds))}
        else:
            rle_segments = collect_rle_window(video_rle_dir)
            track_inds = None
            tubes = get_tubes(rle_segments)

        
        
        # if os.path.exists(os.path.join(video_out_dir, "metrics_STQ.json")):
        #     continue


        gt_ids_vid = [int(k) for k in sample['meta']['class_dict'].keys()]
        for gt_id in gt_ids_vid:
            track_aq_stats[gt_id] = {"size": 0, "tube_stats": {}}
            for tube_id in tubes:
                track_aq_stats[gt_id]["tube_stats"][tube_id] = {"TPA": 0, "FPA": 0, "FNA": 0}

        total_inter = 0
        total_union_non_overlap = 0
        total_union_overlap = 0
        for sample_i, sample in enumerate(tqdm(video, position=1)):
            # Read in image label
            gt_mask = sample['label']['mask'][0]
            gt_ids_frame = np.unique(gt_mask)
            gt_ids_frame = gt_ids_frame[gt_ids_frame!=0]

            # Read in image prediction
            rle_file = os.path.join(video_rle_dir, f"{sample['meta']['window_names'][0].split('.')[0]}.pt")

            pred_masks = metric_utils.read_panomaskRLE(rle_file, inds=track_inds)
            if len(pred_masks)==0:
                pred_masks = torch.zeros((len(tubes.keys()),480,640), dtype=torch.bool)

            obj_mask_bin = torch.as_tensor(gt_mask>0).to(dtype=torch.bool, device=device)
            obj_pred_all = pred_masks.sum(dim=0).to(device=device)
            obj_pred_bin = (obj_pred_all>0).to(dtype=torch.bool, device=device)
            total_inter += (obj_mask_bin*obj_pred_bin).sum().item()
            total_union_overlap += torch.maximum(obj_mask_bin,obj_pred_all).sum().item()
            total_union_non_overlap += torch.logical_or(obj_mask_bin, obj_pred_bin).sum().item()

            # import matplotlib.pyplot as plt
            # fig,ax=plt.subplots(ncols=2)
            # ax[0].imshow(gt_mask)
            # ax[1].imshow(pred_masks.sum(dim=0))
            # plt.show()

            # If a ground truth track was seen in this frame, must calculate tpa,fpa,fna statistics
            for gt_id in gt_ids_frame:
                assert gt_id in track_aq_stats
                gt_mask_bin = torch.as_tensor((gt_mask == gt_id), dtype=torch.bool).to(device=device)
                track_aq_stats[gt_id]["size"] += gt_mask_bin.sum().item()

                for tube_id in tubes:
                    if tubes[tube_id][sample_i]>=0:
                        pred_mask_bin = pred_masks[tubes[tube_id][sample_i]].to(dtype=torch.bool, device=device)

                        tpa = (pred_mask_bin*gt_mask_bin).sum().item()
                        fpa = (pred_mask_bin*(~gt_mask_bin)).sum().item()
                        fna = ((~pred_mask_bin)*gt_mask_bin).sum().item()
                        # print(tpa,fpa,fna)
                        # if tpa>0:
                        #     import matplotlib.pyplot as plt
                        #     fig,ax=plt.subplots(ncols=2)
                        #     ax[0].imshow(pred_mask_bin.cpu())
                        #     ax[1].imshow(gt_mask_bin.cpu())
                        #     plt.show()
                    else:
                        tpa = 0
                        fpa = 0
                        fna = gt_mask_bin.sum().item()

                    track_aq_stats[gt_id]["tube_stats"][tube_id]["TPA"] += tpa
                    track_aq_stats[gt_id]["tube_stats"][tube_id]["FPA"] += fpa
                    track_aq_stats[gt_id]["tube_stats"][tube_id]["FNA"] += fna               


            # If a ground truth track wasn't seen in this frame, only account for fpa statistics
            for gt_id in np.setdiff1d(gt_ids_vid, gt_ids_frame):
                for tube_id in tubes:
                    # Tube is only false positive if it was detected at this frame
                    if tubes[tube_id][sample_i]>=0:
                        pred_mask_bin = pred_masks[tubes[tube_id][sample_i]].to(dtype=torch.bool, device=device)
                        tpa = 0
                        fpa = pred_mask_bin.sum().item()
                        fna = 0
                        track_aq_stats[gt_id]["tube_stats"][tube_id]["FPA"] += fpa

                        

        outer_sum = 0.0
        t = 0
        for gt_id in gt_ids_vid:
            if track_aq_stats[gt_id]["size"]==0:
                del track_aq_stats[gt_id]
                continue
            t+=1
            inner_sum = 0.0
            for tube_id in track_aq_stats[gt_id]["tube_stats"]:
                tpa = track_aq_stats[gt_id]["tube_stats"][tube_id]["TPA"]
                fpa = track_aq_stats[gt_id]["tube_stats"][tube_id]["FPA"]
                fna = track_aq_stats[gt_id]["tube_stats"][tube_id]["FNA"]
                inner_sum += tpa * (tpa / (tpa + fpa + fna))
            outer_sum += (1.0 / track_aq_stats[gt_id]["size"]) * inner_sum
        video_statistics["AQ"] = outer_sum
        video_statistics["inter"] = total_inter
        video_statistics["union_overlap"] = total_union_overlap
        video_statistics["union_non_overlap"] = total_union_non_overlap

        video_statistics["SQ_overlap"] = total_inter / total_union_overlap
        video_statistics["SQ_non_overlap"] = total_inter / total_union_non_overlap
        video_statistics["N_tracks"] = len(list(video_statistics["track_stats"].keys()))
        print(video_statistics["AQ"], t, video_statistics["N_tracks"], video_statistics["AQ"]/video_statistics["N_tracks"])
        print(total_inter, total_union_overlap, total_union_non_overlap)
        print(video_statistics["SQ_overlap"], video_statistics["SQ_non_overlap"])
        print(os.path.join(video_out_dir, "metrics_STQ.json"))
        os.makedirs(video_out_dir, exist_ok=True)
        with open(os.path.join(video_out_dir, "metrics_STQ.json"),"w") as fl:
            json.dump(video_statistics, fl)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--rle_path',type=str, required=True) 
    parser.add_argument('--ref_path',type=str, required=True)
    parser.add_argument('--ref_split',type=str, required=True)
    parser.add_argument('--compute', action='store_true', default=False)
    parser.add_argument('--summarize', action='store_true', default=False)
    args = parser.parse_args()
    

    in_rle_dir = os.path.join(args.rle_path, 'panomasksRLE')
    out_json_dir = os.path.join(args.rle_path, 'metricsJSON')
    


    device = 'cuda'# if torch.cuda.is_available() else 'cpu'

    dataset = MVPDataset(root=args.ref_path,
                    split=args.ref_split,
                    use_stuff=False,
                    window_size = 0)


    if args.compute:
        evaluate_STQ(in_rle_dir, out_json_dir, dataset, device=device)
    if args.summarize:
        summarize_STQ(out_json_dir, fpath="metrics_STQ.json")
