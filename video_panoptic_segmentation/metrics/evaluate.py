import os
import json
import argparse

from tqdm import tqdm

import torch
import numpy as np

from panopticapi.utils import id2rgb

from MVPd.utils.MVPdHelpers import label_to_one_hot
from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from video_panoptic_segmentation.metrics import utils as metric_utils




def get_mask_boundary_metrics(anno_mask, pred_mask):
    anno_boundary = metric_utils.mask_to_boundary(anno_mask.cpu().numpy().astype(np.uint8), dilation_ratio=0.002)
    pred_boundary = metric_utils.mask_to_boundary(pred_mask.cpu().numpy().astype(np.uint8), dilation_ratio=0.002)
    anno_boundary = torch.as_tensor(anno_boundary > 0, device=device)
    pred_boundary = torch.as_tensor(pred_boundary > 0, device=device)

    mask_metrics = {"Mask-"+key: value for key,value in metric_utils.segment_metrics(anno_mask, pred_mask).items()}
    boundary_metrics = {"Boundary-"+key: value for key,value in metric_utils.segment_metrics(anno_boundary, pred_boundary).items()}

    return mask_metrics, boundary_metrics

def evaluate_metrics(in_rle_dir, out_rgb_dir, dataset, device='cpu'):
    results = {}
    for video in tqdm(dataset):
        sample = next(iter(video))
        video_name = sample['meta']['video_name']
        video_rle_dir = os.path.join(in_rle_dir, video_name)
        # video_rgb_dir = os.path.join(out_rgb_dir, video_name)
        # os.makedirs(video_rgb_dir, exist_ok=True)
        results[video_name] = []

        for sample in video:
            sample_result = {}
            window_stamp = '.'.join(next(iter(sample['meta']['window_names'])).split('.')[:-1])
            rle_file = os.path.join(video_rle_dir, window_stamp+'.pt')
            # rgb_file = os.path.join(video_rgb_dir, window_stamp+'.png')

            rle_segments = metric_utils.read_panomaskRLE(rle_file)
            rle_segments = rle_segments.to(dtype=torch.bool).to(device=device)

            ref_arr = sample['label']['mask'][0]
            ref_segments, ref_ids = label_to_one_hot(ref_arr, filter_void=True)            
            ref_segments = torch.as_tensor(ref_segments, device=device, dtype=torch.bool)

            # Match rle segments to reference segments, then merge unmatched rle segments
            (matched_ref_ind, unmatched_ref_ind), (matched_rle_ind, unmatched_rle_ind), _ = metric_utils.match_segments(ref_segments, rle_segments)
            
            # ref_rgbs = np.array(id2rgb(ref_ids))

            # rle_rgbs = np.zeros((rle_segments.shape[0],4))
            # # print(ref_rgbs[matched_ref_ind].shape, np.array(matched_ref_ind.shape[0]*[[255]]).shape)
            # if ref_rgbs[matched_ref_ind].shape[0]>0:
            #     rle_rgbs[matched_rle_ind] = np.concatenate([ref_rgbs[matched_ref_ind], matched_ref_ind.shape[0]*[[255]]],axis=1)            
            # rle_rgbs[unmatched_rle_ind] = (255,255,255,0.65*255)

            # for unmatched_rle_i in unmatched_rle_ind:
            #     rle_segments[unmatched_rle_i] = torch.as_tensor(metric_utils.mask_to_boundary(rle_segments[unmatched_rle_i].cpu().float().numpy(), 
            #         dilation_ratio=0.004))

            # rle_panomaskrgb = Image.fromarray(metric_utils.binmasks_to_panomask(rle_segments.cpu().numpy(), rle_rgbs).astype(np.uint8))
            # rle_panomaskrgb.save(rgb_file)

            for ref_ind, rle_ind in zip(matched_ref_ind, matched_rle_ind):
                anno_mask, pred_mask = ref_segments[ref_ind], rle_segments[rle_ind]

                mask_metrics, boundary_metrics = get_mask_boundary_metrics(anno_mask, pred_mask)

                sample_result[int(rle_ind)] = {**mask_metrics, **boundary_metrics}
                sample_result[int(rle_ind)]["instance_id"] = int(ref_ids[ref_ind])
                sample_result[int(rle_ind)]["category_id"] = int(sample['meta']['class_dict'][int(ref_ids[ref_ind])])

            for rle_ind in unmatched_rle_ind:
                pred_mask = rle_segments[rle_ind]
                anno_mask = torch.zeros_like(pred_mask)

                mask_metrics, boundary_metrics = get_mask_boundary_metrics(anno_mask, pred_mask)

                sample_result[int(rle_ind)] = {**mask_metrics, **boundary_metrics}
            
            for ref_ind in unmatched_ref_ind:
                anno_mask = ref_segments[ref_ind]
                pred_mask = torch.zeros_like(anno_mask)

                mask_metrics, boundary_metrics = get_mask_boundary_metrics(anno_mask, pred_mask)
                
                sample_result[f'unmatched-anno-{int(ref_ind)}'] = {**mask_metrics, **boundary_metrics}
                sample_result[f'unmatched-anno-{int(ref_ind)}']["instance_id"] = int(ref_ids[ref_ind])
                sample_result[f'unmatched-anno-{int(ref_ind)}']["category_id"] = int(sample['meta']['class_dict'][int(ref_ids[ref_ind])])

            results[video_name].append(sample_result)

    return results

# def evaluate_metrics(anno_dir_seqs, pred_dir_seqs, dataset):
#     anno_dir, anno_sequences = anno_dir_seqs
#     pred_dir, pred_sequences = pred_dir_seqs

#     assert len(anno_sequences)==len(pred_sequences)
#     assert set(anno_sequences)==set(pred_sequences)

#     results = {}

#     for sequence in tqdm(anno_sequences):
#         annotations = next(v for v in dataset["annotations"] if v["video_name"]==sequence)["annotations"]

#         results[sequence] = []

#         for anno in annotations:

#             anno_result = {}

#             file_path = os.path.join(sequence, anno["file_name"])

#             anno_file_path = os.path.join(anno_dir, file_path)
#             pred_file_path = os.path.join(pred_dir, file_path)

#             if not (os.path.isfile(anno_file_path) and os.path.isfile(pred_file_path)):
#                 continue

#             anno_arr = metric_utils.read_panomaskRGB(anno_file_path)
#             pred_arr = metric_utils.read_panomaskRGB(pred_file_path)

#             assert set(np.unique(anno_arr)).difference(set([segment["id"] for segment in anno["segments_info"]]))=={0}
            

#             for segment_id in np.sort(np.unique(anno_arr)):
#                 if segment_id!=0:
#                     segment = next(seg for seg in anno["segments_info"] if seg["id"]==segment_id)
#                     category_id = segment["category_id"]
#                 else:
#                     category_id = 0


#                 anno_mask = anno_arr==segment_id
#                 pred_mask = pred_arr==segment_id

                
                
#                 anno_boundary = metric_utils.mask_to_boundary(anno_mask.astype(np.uint8), dilation_ratio=0.002)
#                 pred_boundary = metric_utils.mask_to_boundary(pred_mask.astype(np.uint8), dilation_ratio=0.002)
#                 anno_boundary = anno_boundary > 0
#                 pred_boundary = pred_boundary > 0


#                 mask_metrics = {"Mask-"+key: value for key,value in metric_utils.segment_metrics(anno_mask, pred_mask).items()}
#                 boundary_metrics = {"Boundary-"+key: value for key,value in metric_utils.segment_metrics(anno_boundary, pred_boundary).items()}

#                 anno_result[int(segment_id)] = {**mask_metrics, **boundary_metrics}
#                 anno_result[int(segment_id)]["category_id"] = int(category_id)

#             results[sequence].append(anno_result)


#     return results



if __name__=='__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--true_path',type=str, default='./VIPOSeg/valid')
    # parser.add_argument('--pred_path',type=str, required=True)
    # args = parser.parse_args()

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
    # rle_2_rgb(in_rle_dir, out_rgb_dir, MVPd, device=device)


    result_dict = evaluate_metrics(in_rle_dir, out_rgb_dir, MVPd, device=device)

    with open(os.path.join(args.pred_path, "metrics.json"),"w") as fl:
        json.dump(result_dict, fl)