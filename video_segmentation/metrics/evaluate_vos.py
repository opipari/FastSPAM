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

from MVPd.utils.MVPdHelpers import label_to_one_hot, filter_idmask_area
from panopticapi.utils import rgb2id
from video_segmentation.metrics import utils as metric_utils

import pickle

from PIL import Image




#
# https://github.com/LingyiHongfd/lvos-evaluation/blob/main/lvos/metrics.py
#

import math
import warnings

import cv2
import numpy as np
from skimage.morphology import disk


def db_eval_iou(annotation, segmentation, void_pixels=None):
    """ Compute region similarity as the Jaccard Index.
    Arguments:
        annotation   (ndarray): binary annotation   map.
        segmentation (ndarray): binary segmentation map.
        void_pixels  (ndarray): optional mask with void pixels

    Return:
        jaccard (float): region similarity
    """
    assert annotation.shape == segmentation.shape, \
        f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape, \
            f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation)

    # Intersection between all sets
    inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
    union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))

    j = inters / union
    if j.ndim == 0:
        j = 1 if np.isclose(union, 0) else j
    else:
        j[np.isclose(union, 0)] = 1
    return j

def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 3:
        n_frames = annotation.shape[0]
        f_res = np.zeros(n_frames)
        for frame_id in range(n_frames):
            void_pixels_frame = None if void_pixels is None else void_pixels[frame_id, :, :, ]
            f_res[frame_id] = f_measure(segmentation[frame_id, :, :, ], annotation[frame_id, :, :], void_pixels_frame, bound_th=bound_th)
    elif annotation.ndim == 2:
        f_res = f_measure(segmentation, annotation, void_pixels, bound_th=bound_th)
    else:
        raise ValueError(f'db_eval_boundary does not support tensors with {annotation.ndim} dimensions')
    return f_res

def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    """
    Compute mean,recall and decay from per-frame evaluation.
    Calculates precision/recall for boundaries between foreground_mask and
    gt_mask using morphological operators to speed it up.

    Arguments:
        foreground_mask (ndarray): binary segmentation image.
        gt_mask         (ndarray): binary annotated image.
        void_pixels     (ndarray): optional mask with void pixels

    Returns:
        F (float): boundaries F-measure
    """
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask).astype(bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    # Get the pixel boundaries of both masks
    fg_boundary = _seg2bmap(foreground_mask * np.logical_not(void_pixels))
    gt_boundary = _seg2bmap(gt_mask * np.logical_not(void_pixels))


    # fg_dil = binary_dilation(fg_boundary, disk(bound_pix))
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
    # gt_dil = binary_dilation(gt_boundary, disk(bound_pix))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))

    # Get the intersection
    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    # Area of the intersection
    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    # % Compute precision and recall
    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    # Compute F measure
    if precision + recall == 0:
        F = 0
    else:
        F = 2 * precision * recall / (precision + recall)

    return F


def _seg2bmap(seg, width=None, height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width     : Width of desired bmap  <= seg.shape[1]
        height  :   Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray): Binary boundary map.
     David Martin <dmartin@eecs.berkeley.edu>
     January 2003
    """

    seg = seg.astype(bool)
    seg[seg > 0] = 1

    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height

    h, w = seg.shape[:2]

    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)

    assert not (
        width > w | height > h | abs(ar1 - ar2) > 0.01
    ), "Can" "t convert %dx%d seg to %dx%d bmap." % (w, h, width, height)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height, width))
        for x in range(w):
            for y in range(h):
                if b[y, x]:
                    j = 1 + math.floor((y - 1) + height / h)
                    i = 1 + math.floor((x - 1) + width / h)
                    bmap[j, i] = 1

    return bmap


def db_statistics(per_frame_values):
    """ Compute mean,recall and decay from per-frame evaluation.
    Arguments:
        per_frame_values (ndarray): per-frame evaluation

    Returns:
        M,O,D (float,float,float):
            return evaluation statistics: mean,recall,decay.
    """

    # strip off nan values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        M = np.nanmean(per_frame_values)
        O = np.nanmean(per_frame_values > 0.5)

    N_bins = 4
    ids = np.round(np.linspace(1, len(per_frame_values), N_bins + 1) + 1e-10) - 1
    ids = ids.astype(int)

    D_bins = [per_frame_values[ids[i]:ids[i + 1] + 1] for i in range(0, 4)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        D = np.nanmean(D_bins[0]) - np.nanmean(D_bins[3])


    return M, O, D



def collect_ref(ref_file, obj_id):

    label_ = Image.open(ref_file)
    label_ = np.array(label_, dtype=np.uint8)
    label_ = rgb2id(label_)
    label_ = filter_idmask_area(label_)
    label_ = label_ == obj_id
    return label_


def collect_rle(rle_file, inds=None, default_size=(480,640)):
    rle_seg = metric_utils.read_panomaskRLE(rle_file, inds=inds)
    if len(rle_seg)==0:
        rle_seg = np.zeros(default_size, dtype=bool)
    return np.asarray(rle_seg.to(torch.bool)[0], dtype=bool)



def evaluate_vos(in_rle_dir, out_dir, ref_path, ref_split, device='cpu'):
    
    video_directory = os.path.join(ref_path, ref_split, 'panomasksRGB')
    pkl_directory = os.path.join(ref_path, ref_split, 'objectsPKL')

    empty_split = json.load(open(os.path.join(ref_path, f'{ref_split}/empty_videos.json'), 'r'))
    videos = [vid for vid in sorted(os.listdir(in_rle_dir)) if vid in empty_split['accepted']]

    for video_name in tqdm(videos):
        video_ref_dir = os.path.join(video_directory, video_name)
        pkl_ref_path = os.path.join(pkl_directory, video_name+'.pkl')

        video_rle_dir = os.path.join(in_rle_dir, video_name)
        video_out_dir = os.path.join(out_dir, video_name)
        
        if os.path.exists(os.path.join(video_out_dir, "metrics_vos.json")):
            continue

        with open(pkl_ref_path, 'rb') as fp:
            ids = pickle.load(fp)
        assert len(ids)==1


        video_results = {"J": [], "F": []}

        for frame_name in sorted(os.listdir(video_rle_dir)):
        
            rle_mask = collect_rle(os.path.join(video_rle_dir, frame_name))
            ref_mask = collect_ref(os.path.join(video_ref_dir, frame_name.split('.')[0]+'.png'), ids[0])
            

            video_results["J"].append(db_eval_iou(ref_mask, rle_mask))
            video_results["F"].append(db_eval_boundary(ref_mask, rle_mask))

        [JM, JR, JD] = db_statistics(np.asarray(video_results["J"]))
        video_results["J"] = {"M": JM, "R": JR, "D": JD}

        [FM, FR, FD] = db_statistics(np.asarray(video_results["F"]))
        video_results["F"] = {"M": FM, "R": FR, "D": FD}

        os.makedirs(video_out_dir, exist_ok=True)
        with open(os.path.join(video_out_dir, "metrics_vos.json"),"w") as fl:
            json.dump(video_results, fl)
        
    


def summarize_vos(metric_root, fpath="metrics_iVPQ.json"):
    metrics = ["JM", "JR", "JD", "FM", "FR", "FD"]
    total = {met: [] for met in metrics}
    v_tot = 0
    for video_name in sorted(os.listdir(metric_root)):
        video_data = json.load(open(os.path.join(metric_root, video_name, fpath),"r"))
            
        for met in metrics:
            total[met].append(video_data[met[0]][met[1]])

        v_tot+=1
        # print()
    print(v_tot)
    for met in metrics:
        print(met, sum(total[met])/len(total[met]))





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

    if args.compute:
        evaluate_vos(in_rle_dir, out_json_dir, args.ref_path, args.ref_split, device)
    if args.summarize:
        summarize_vos(out_json_dir, fpath="metrics_vos.json")
