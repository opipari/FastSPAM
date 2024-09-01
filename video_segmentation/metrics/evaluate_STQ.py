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




def calc_aq_score(num_tracks, num_preds, get_track, get_pred, get_track_size):

    score = 0
    for pred_i in tqdm(range(num_preds)):
        pred_mask_bin = get_pred(pred_i).to(torch.bool) # T x H x W
        
        for gt_j in range(num_tracks):
            gt_mask_bin = get_track(gt_j).to(torch.bool) # T x H xW
            
            tpa = (gt_mask_bin * pred_mask_bin).sum().item()
            fpa = ((~gt_mask_bin) * pred_mask_bin).sum().item()
            fna = (gt_mask_bin * (~pred_mask_bin)).sum().item()

            score += (1 / get_track_size(gt_j)) * tpa * (tpa / (tpa + fpa + fna))
    
    return score


def calc_sq_score(gt_masks_all_bin, pred_masks_all_bin):
    assert gt_masks_all_bin.shape==pred_masks_all_bin.shape
    num_classes = gt_masks_all_bin.shape[0]

    class_i = []
    class_u = []
    for c in range(num_classes):
        if c==(num_classes-1):
            class_i.append(torch.tensor((0)))
            # Remove pixels labeled as background from false positive calculation
            class_u.append(torch.logical_and(pred_masks_all_bin[c], ~gt_masks_all_bin[-1]).sum())
        else:
            class_i.append(torch.logical_and(gt_masks_all_bin[c], pred_masks_all_bin[c]).sum())
            # Remove pixels labeled as background from false positive calculation
            class_u.append(torch.logical_or(gt_masks_all_bin[c], torch.logical_and(pred_masks_all_bin[c], ~gt_masks_all_bin[-1])).sum())

    return torch.as_tensor(class_i), torch.as_tensor(class_u)








def collect_rle_tracks(video, video_rle_dir, threshold=0.5):
    track_inds = set()

    for f_idx in range(len(video)):
        rle_file = os.path.join(video_rle_dir, f'{f_idx:010d}.pt')
        track_boxes = torch.load(rle_file, map_location='cpu')['bbox_results']
        track_inds.update(np.flatnonzero(track_boxes[:,5]>threshold))

    return list(track_inds)


def collect_rle_window(video_rle_dir, track_inds=None, default_size=(1,480,640)):
    rle_segments = []
    for name in sorted(os.listdir(video_rle_dir)):
        rle_file = os.path.join(video_rle_dir, name)
        rle_seg = metric_utils.read_panomaskRLE(rle_file, inds=track_inds)
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




def masks_to_sem_tubes(labels, preds, ign_id=0):

    sem_label_tubes = torch.stack([labels!=ign_id, labels==ign_id]) # C x T x H x W
    sem_pred_tubes = torch.stack([p.sum(dim=0)>0 for p in preds]) # T x H x W
    sem_pred_tubes = torch.stack([sem_pred_tubes, ~sem_pred_tubes]) # C x T x H x W

    sem_label_tubes = sem_label_tubes.to(torch.bool)
    sem_pred_tubes = sem_pred_tubes.to(torch.bool)

    return sem_label_tubes, sem_pred_tubes



def evaluate_STQ(in_rle_dir, dataset, epsilon=1e-15):

    # torch.random.manual_seed(0)
    # torch.randperm(1741)[:10]
    rand_inds = [1367,  100,  983,   83, 1586,  896,  683, 1598, 1092, 1020]


    num_videos = len(rand_inds)
    num_classes = 1

    aq_per_seq = torch.zeros((num_videos))
    num_tubes_per_seq = torch.zeros((num_videos))
    iou_per_seq = torch.zeros((num_videos))
    stq_per_seq = torch.zeros((num_videos))

    class_intersctions = torch.zeros((num_classes+1))
    class_unions = torch.zeros((num_classes+1))


    # for vii,video in enumerate(tqdm(dataset, position=0)):
    for vid_id, video_i in enumerate(rand_inds):
        video = dataset[video_i]
        sample = next(iter(video))
        video_name = sample['meta']['video_name']
        video_rle_dir = os.path.join(in_rle_dir, video_name)
        # video_out_dir = os.path.join(out_dir, video_name)

        labels = []
        for data in video:
            labels.append(torch.as_tensor(data['label']['mask'][0]))
        labels = torch.stack(labels) # T x H x W
        assert labels.shape[0]==len(video)

        sample_rle_file = os.path.join(video_rle_dir, sample['meta']['window_names'][0].split('.')[0]+'.pt')
        sample_rle = torch.load(sample_rle_file,map_location='cpu')
        if 'bbox_results' in sample_rle:
            track_inds = collect_rle_tracks(video, video_rle_dir)
            preds = collect_rle_window(video_rle_dir, track_inds=track_inds)
            tubes = {k: [k]*len(video) for k in range(len(track_inds))}
        else:
            track_inds = None
            preds = collect_rle_window(video_rle_dir)
            tubes = get_tubes(preds)
            for k in tubes:
                assert len(tubes[k])==len(video)
        print(len(tubes.keys()))
        # Calculate SQ
        sem_label_tubes, sem_pred_tubes = masks_to_sem_tubes(labels, preds)
        assert sem_label_tubes.shape==sem_pred_tubes.shape

        intersections, unions = calc_sq_score(sem_label_tubes, sem_pred_tubes)
        class_intersctions += intersections
        class_unions += unions

        intersections = intersections[unions>0]
        unions = unions[unions>0]
        
        del sem_label_tubes
        del sem_pred_tubes

        sq_ = torch.mean(intersections / torch.clamp(unions, min=epsilon))
        print(vid_id, sq_)

        # Calculate AQ
        if 'bbox_results' in sample_rle:
            preds = torch.stack(preds).permute(1,0,2,3)
            get_pred = lambda pred_i: preds[pred_i]
        else:
            get_pred = lambda pred_i: torch.stack([preds[t][tubes[pred_i][t]] if tubes[pred_i][t]>=0 else torch.zeros((480,640)) for t in range(len(video))])
        
        label_ids = torch.unique(labels)
        label_ids = label_ids[label_ids!=0]
        
        num_preds = len(tubes.keys())
        num_tracks = label_ids.shape[0]

        get_track = lambda gt_j: labels==label_ids[gt_j]
        inst_label_sizes = torch.as_tensor([get_track(gt_j).sum() for gt_j in range(num_tracks)])
        get_track_size = lambda gt_j: inst_label_sizes[gt_j]

        aq_score = calc_aq_score(num_tracks, num_preds, get_track, get_pred, get_track_size)
        aq_per_seq[vid_id] = aq_score
        num_tubes_per_seq[vid_id] = num_tracks

        del labels
        del preds

        
        aq_ = aq_per_seq[vid_id] / torch.clamp(num_tubes_per_seq[vid_id], min=epsilon)

        stq_per_seq[vid_id] = torch.sqrt(aq_*sq_)
        iou_per_seq[vid_id] = sq_
    
        aq_mean = aq_per_seq.sum() / torch.clamp(num_tubes_per_seq.sum(), min=epsilon)
        num_classes_nonzero = len(class_unions.nonzero())
        ious = class_intersctions / torch.clamp(class_unions, min=epsilon)
        iou_mean =  ious.sum() / num_classes_nonzero

        stq = torch.sqrt(aq_mean * iou_mean)
        print(f"{vid_id} stq:{stq.item()}, stq_vid:{stq_per_seq[vid_id].item()}, aq_vid:{aq_.item()}, sq_vid:{sq_.item()}")

        torch.save({
            "stq": stq.item(),
            "aq": aq_mean.item(),
            "iou": iou_mean.item(),
            "aq_per_seq": aq_per_seq,
            "num_tubes_per_seq": num_tubes_per_seq,
            "iou_per_seq": iou_per_seq,
            "stq_per_seq": stq_per_seq,
            "class_intersections": class_intersctions,
            "class_unions": class_unions
            }, "res.json")




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
        evaluate_STQ(in_rle_dir, dataset)
    if args.summarize:
        summarize_STQ(out_json_dir, fpath="metrics_STQ.json")
