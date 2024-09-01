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
from video_segmentation.metrics.evaluate_STQ import calc_aq_score, calc_sq_score



def collect_masks(video_dir, vid_id):
    gt_video_dir = os.path.join(video_dir, 'gt')
    pred_video_dir = os.path.join(video_dir, 'pred')
    assert len(os.listdir(gt_video_dir))==len(os.listdir(pred_video_dir))
    file_list = sorted([fl for fl in os.listdir(gt_video_dir) if fl.startswith(f'{vid_id:06d}')])

    labels = []
    preds = []
    for file in file_list:
        label_file = os.path.join(gt_video_dir, file)
        pred_file = os.path.join(pred_video_dir, file)
        
        label_arr = torch.load(label_file)
        pred_arr = torch.load(pred_file)
        
        labels.append(label_arr)
        preds.append(pred_arr)
    labels = np.stack(labels)
    preds = np.stack(preds)

    labels = labels.astype(np.int64)
    preds = preds.astype(np.int64)

    return labels, preds

def masks_to_sem_tubes(labels, preds, num_classes=124, max_ins=10000, ign_id=255, num_things=58, label_divisor=1e4, ins_divisor=1e7):

    sem_labels = labels // max_ins
    sem_preds = preds // max_ins
    # print(sem_labels.min(), sem_preds.min())
    # print(sem_labels.max(),sem_labels[sem_labels!=255].max(), sem_preds.max())
    sem_labels = np.where(sem_labels != ign_id,
                        sem_labels, num_classes)
    sem_preds = np.where(sem_preds != ign_id,
                        sem_preds, num_classes)

    sem_label_tubes = np.expand_dims(sem_labels, axis=1) == np.arange(num_classes+1).astype(sem_labels.dtype).reshape(1,num_classes+1,1,1)
    sem_pred_tubes = np.expand_dims(sem_preds, axis=1) == np.arange(num_classes+1).astype(sem_preds.dtype).reshape(1,num_classes+1,1,1)

    sem_label_tubes = torch.as_tensor(sem_label_tubes).to(torch.bool)
    sem_pred_tubes = torch.as_tensor(sem_pred_tubes).to(torch.bool)

    sem_label_tubes = sem_label_tubes.permute(1,0,2,3)      # C x T x H x W
    sem_pred_tubes = sem_pred_tubes.permute(1,0,2,3)        # C x T x H x W

    return sem_label_tubes, sem_pred_tubes


def masks_to_inst_tubes(labels, preds, num_classes=124, max_ins=10000, ign_id=255, num_things=58, label_divisor=1e4, ins_divisor=1e7):
    sem_labels = labels // max_ins
    sem_labels = np.where(sem_labels != ign_id,
                        sem_labels, num_classes)

    instance_labels = labels % max_ins
    label_masks = np.less(sem_labels, num_things)
    is_crowd = np.logical_and(instance_labels == 0, label_masks)

    inst_masks = np.logical_and(label_masks, np.logical_not(is_crowd))

    inst_labels = labels * inst_masks
    inst_preds = preds * inst_masks

    inst_label_tubes, _ = label_to_one_hot(inst_labels, filter_void=True)
    inst_pred_tubes, _ = label_to_one_hot(inst_preds, filter_void=True)

    inst_label_tubes = torch.as_tensor(inst_label_tubes).to(torch.bool)
    inst_pred_tubes = torch.as_tensor(inst_pred_tubes).to(torch.bool)

    inst_label_tubes = inst_label_tubes.permute(1,0,2,3)    # L x T x H x W
    inst_pred_tubes = inst_pred_tubes.permute(1,0,2,3)    # P x T x H x W

    return inst_label_tubes, inst_pred_tubes



def evaluate_STQ(path, epsilon=1e-15):

    num_videos = 343
    num_classes = 124

    aq_per_seq = torch.zeros((num_videos))
    num_tubes_per_seq = torch.zeros((num_videos))
    iou_per_seq = torch.zeros((num_videos))
    stq_per_seq = torch.zeros((num_videos))

    class_intersctions = torch.zeros((num_classes+1))
    class_unions = torch.zeros((num_classes+1))

    for vid_id in tqdm(range(343)):
        labels, preds = collect_masks(path, vid_id)
        
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

        # Calculate AQ
        inst_label_tubes, inst_pred_tubes = masks_to_inst_tubes(labels, preds)
        assert inst_label_tubes.shape[1:]==inst_pred_tubes.shape[1:]

        inst_pred_sizes = inst_pred_tubes.sum((1,2,3))
        inst_pred_tubes = inst_pred_tubes[inst_pred_sizes>0]

        inst_label_sizes = inst_label_tubes.sum((1,2,3))
        inst_label_tubes = inst_label_tubes[inst_label_sizes>0]
        inst_label_sizes = inst_label_sizes[inst_label_sizes>0]
        
        num_preds = inst_pred_tubes.shape[0]
        num_tracks = inst_label_tubes.shape[0]

        get_pred = lambda pred_i: inst_pred_tubes[pred_i]
        get_track = lambda gt_j: inst_label_tubes[gt_j]
        get_track_size = lambda gt_j: inst_label_sizes[gt_j]

        aq_score = calc_aq_score(num_tracks, num_preds, get_track, get_pred, get_track_size)
        aq_per_seq[vid_id] = aq_score
        num_tubes_per_seq[vid_id] = num_tracks

        del inst_label_tubes
        del inst_pred_tubes

        sq_ = torch.mean(intersections / torch.clamp(unions, min=epsilon))
        aq_ = aq_per_seq[vid_id] / torch.clamp(num_tubes_per_seq[vid_id], min=epsilon)

        stq_per_seq[vid_id] = torch.sqrt(aq_*sq_)
        iou_per_seq[vid_id] = sq_
        print(stq_per_seq[vid_id].item(), aq_.item(), sq_.item())
    
    aq_mean = aq_per_seq.sum() / torch.clamp(num_tubes_per_seq.sum(), min=epsilon)
    num_classes_nonzero = len(class_unions.nonzero())
    ious = class_intersctions / torch.clamp(class_unions, min=epsilon)
    iou_mean =  ious.sum() / num_classes_nonzero

    stq = torch.sqrt(aq_mean * iou_mean)

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

# def evaluate_STQ(in_rle_dir, out_dir, dataset, conf_threshold=0.5, device='cpu'):
    
    
#     # for vii,video in enumerate(tqdm(dataset, position=0)):
#     for vii, video_i in enumerate(rand_inds):
#         video = dataset[video_i]
#         sample = next(iter(video))
#         video_name = sample['meta']['video_name']
#         video_rle_dir = os.path.join(in_rle_dir, video_name)
#         video_out_dir = os.path.join(out_dir, video_name)

#         video_statistics = {
#             "AQ": 0.0, 
#             "SQ_overlap": 0.0, 
#             "SQ_non_overlap":0.0, 
#             "N_tracks": 0.0, 
#             "track_stats": {}}
#         track_aq_stats = video_statistics["track_stats"]

        
#         sample_rle_file = os.path.join(video_rle_dir, sample['meta']['window_names'][0].split('.')[0]+'.pt')
#         sample_rle = torch.load(sample_rle_file,map_location='cpu')
#         if 'bbox_results' in sample_rle:
#             track_inds = collect_rle_tracks(video, video_rle_dir)
#             tubes = {k: [k]*len(video) for k in range(len(track_inds))}
#         else:
#             rle_segments = collect_rle_window(video_rle_dir)
#             track_inds = None
#             tubes = get_tubes(rle_segments)

        
        
#         # if os.path.exists(os.path.join(video_out_dir, "metrics_STQ.json")):
#         #     continue


#         gt_ids_vid = [int(k) for k in sample['meta']['class_dict'].keys()]
#         for gt_id in gt_ids_vid:
#             track_aq_stats[gt_id] = {"size": 0, "tube_stats": {}}
#             for tube_id in tubes:
#                 track_aq_stats[gt_id]["tube_stats"][tube_id] = {"TPA": 0, "FPA": 0, "FNA": 0}

#         all_gt_masks = []
#         for sample_i, sample in enumerate(tqdm(video, position=1)):
#             gt_mask = sample['label']['mask'][0]
#             all_gt_masks.append(torch.as_tensor(gt_mask))
#         all_gt_masks = torch.stack(all_gt_masks)
#         print("Ground Truth:", all_gt_masks.shape) # T x H x W

#         # Read in all of the 
#         pred_masks = collect_rle_window(video_rle_dir, track_inds=track_inds)
#         pred_masks = torch.stack(pred_masks)
#         print(pred_masks.dtype)
#         num_preds = pred_masks.shape[1]
#         print(num_preds)
#         print("Predictions:",pred_masks.shape)

#         gt_ids_vid = torch.unique(all_gt_masks)
#         gt_ids_vid = gt_ids_vid[gt_ids_vid!=0]

#         boundaries = range(0,num_preds)
#         # total = 0
#         # for ba,bb in zip(boundaries,boundaries[1:]):
#         #     print(ba,bb)
#         #     score = sum_func(ba,bb)
#         #     print("AQ(i):", score)
#         #     total += score
#         final_score = sum_func(boundaries[0],boundaries[-1])
#         print("AQ(final):", final_score) 
#         # print("AQ(total):", total)
#         # print("Correct:",abs(total-final_score))
#         # print()
        
            
#         # 17905153.78619084
#         raise



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path',type=str, required=True) 
    parser.add_argument('--compute', action='store_true', default=False)
    # parser.add_argument('--summarize', action='store_true', default=False)
    args = parser.parse_args()
    

    # in_rle_dir = os.path.join(args.rle_path, 'panomasksRLE')
    # out_json_dir = os.path.join(args.rle_path, 'metricsJSON')
    


    # device = 'cpu'# if torch.cuda.is_available() else 'cpu'



    if args.compute:
        evaluate_STQ(args.path)
    # if args.summarize:
    #     summarize_STQ(out_json_dir, fpath="metrics_STQ.json")
