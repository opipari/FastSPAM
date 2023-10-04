from typing import Tuple

import cv2
import numpy as np

import torch
import scipy

from PIL import Image

from panopticapi.utils import rgb2id
from pycocotools import mask as mask_utils

from typing import Any, Dict, Generator, ItemsView, List, Tuple




def read_panomaskRGB(
    path: str
) -> np.ndarray:
    label = Image.open(path)
    label = np.array(label, dtype=np.uint8)
    label = rgb2id(label)
    return label

def read_panomaskRLE(
    pt_file_path: str
) -> torch.Tensor:
    mask_list = [torch.as_tensor(mask_utils.decode(rle),dtype=torch.int) for rle in torch.load(pt_file_path,map_location='cpu')['coco_rle']]
    if len(mask_list)>0:
        return torch.stack(mask_list)
    else:
        return mask_list

def alpha_compose(
    nsrc,
    ndst
):
    # Based on: https://stackoverflow.com/questions/60398939/how-to-do-alpha-compositing-with-a-list-of-rgba-data-in-numpy-arrays

    # Extract the RGB channels
    srcRGB = nsrc[...,:3]
    dstRGB = ndst[...,:3]

    # Extract the alpha channels and normalise to range 0..1
    srcA = nsrc[...,3]/255.0
    dstA = ndst[...,3]/255.0

    # Work out resultant alpha channel
    outA = srcA + dstA*(1-srcA)

    # Work out resultant RGB
    outRGB = (srcRGB*srcA[...,np.newaxis] + dstRGB*dstA[...,np.newaxis]*(1-srcA[...,np.newaxis])) / outA[...,np.newaxis]

    # Merge RGB and alpha (scaled back up to 0..255) back into single image
    outRGBA = np.dstack((outRGB,outA*255)).astype(np.uint8)
    return outRGBA


def binmasks_to_panomask(
    binmasks: np.ndarray,
    colors: np.ndarray
) -> np.ndarray:
    _, h, w = binmasks.shape

    panomask = np.zeros((h,w,colors.shape[1]))
    if panomask.shape[2]==4:
        panomask[:,:,3]=255
    if len(binmasks) > 0:
        sorted_indices = np.argsort([int(mask.sum()) for mask in binmasks])[::-1]
        for sort_idx in sorted_indices:
            if panomask.shape[2]==4:
                panomask = alpha_compose(np.expand_dims(binmasks[sort_idx],2)*colors[sort_idx], panomask)
            else:
                panomask[binmasks[sort_idx]] = colors[sort_idx]

    return panomask



def mask_to_boundary(
    mask: np.ndarray,
    dilation_ratio: float = 0.02
) -> np.ndarray:
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def segment_metrics(
    true_segment: torch.Tensor,
    pred_segment: torch.Tensor
) -> dict:
    """Calculate pairwise pixel-wise metrics (precision, recall, f-score, iou) between two segments.

    Keyword arguments:
    true_segment -- binary tensor of shape (H,W)
    pred_segment -- binary tensor of shape (H,W)
    eps -- float to avoid numerical overflow for division
    """
    assert true_segment.dtype == pred_segment.dtype == torch.bool
    assert true_segment.shape == pred_segment.shape

    intersection = ((true_segment * pred_segment) > 0).sum().cpu().item()
    precision_denominator = (pred_segment>0).sum().cpu().item()
    recall_denominator = (true_segment>0).sum().cpu().item()

    precision = intersection / precision_denominator if precision_denominator > 0 else 0
    recall = intersection / recall_denominator if recall_denominator > 0 else 0
    F_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    IOU_score = intersection / (precision_denominator + recall_denominator - intersection) if (precision_denominator + recall_denominator - intersection) > 0 else 0


    results = {"Precision": float(precision),
                "Recall": float(recall),
                "F-Score": float(F_score),
                "IOU": float(IOU_score),
                "Intersection": float(intersection),
                "Precision-Denominator": float(precision_denominator),
                "Recall-Denominator": float(recall_denominator)
                }
    return results


def pairwise_segment_metric(
    true_segments: torch.Tensor,
    pred_segments: torch.Tensor, 
    metric: str = 'F-Score'
) -> torch.Tensor:
    """Calculate pairwise pixel-wise metrics (precision, recall, f-score, iou) between two sets of segments.

    Keyword arguments:
    true_segments -- binary tensor of shape (A,H,W) where A corresponds to number of true segments in a
    pred_segments -- binary tensor of shape (B,H,W) where B corresponds to number of predicted segments in b
    metric -- string specifying which segmentation metric to calculate and return
    """
    assert true_segments.ndim == pred_segments.ndim == 3
    assert true_segments.shape[1:] == pred_segments.shape[1:]
    num_true_segments, num_pred_segments = true_segments.shape[0], pred_segments.shape[0]
        
    pairwise_metric = torch.zeros((num_true_segments, num_pred_segments))
    for idx_a in range(num_true_segments):
        for idx_b in range(num_pred_segments):
            pairwise_metric[idx_a,idx_b] = segment_metrics(true_segments[idx_a], pred_segments[idx_b])[metric]

    return pairwise_metric


def match_segments(
    true_segments: torch.Tensor,
    pred_segments: torch.Tensor, 
    metric: str = 'F-Score'
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:
    """Calculate matched segments given a two segmentation candidates on a single view.

    Keyword arguments:
    predicted_segments -- binary tensor of shape (A,H,W) where A corresponds to number of predicted segments in a
    labeled_segments -- binary tensor of shape (B,H,W) where B corresponds to number of predicted segments in b
    eps -- float to avoid numerical overflow for division
    """

    pairwise_metric = pairwise_segment_metric(true_segments, pred_segments, metric=metric)

    matched_true_ind, matched_pred_ind = scipy.optimize.linear_sum_assignment(pairwise_metric.cpu().numpy(), maximize=True)

    # Account for the possibility that linear sum may assign predicted segments to true segments with no overlap
    # In this case, remove the assignment from consideration to allow for downstreat merge of prediction into better assignement
    no_overlap_inds = [idx for idx in range(len(matched_pred_ind)) if pairwise_metric[matched_true_ind[idx],matched_pred_ind[idx]]==0]

    matched_true_ind, matched_pred_ind = np.delete(matched_true_ind, no_overlap_inds), np.delete(matched_pred_ind, no_overlap_inds)

    unmatched_true_ind = np.setdiff1d(np.arange(true_segments.shape[0]), matched_true_ind, assume_unique=True)
    unmatched_pred_ind = np.setdiff1d(np.arange(pred_segments.shape[0]), matched_pred_ind, assume_unique=True)
   
    return (matched_true_ind, unmatched_true_ind), (matched_pred_ind, unmatched_pred_ind), pairwise_metric


def merge_unmatched_segments(
    pred_segments: torch.Tensor,
    matched_indices: np.ndarray,
    metric: str = 'F-Score'
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor]:

    unmatched_indices = np.setdiff1d(np.arange(pred_segments.shape[0]), matched_indices, assume_unique=True)

    matched_segments_merged = pred_segments[matched_indices].clone()
    unmatched_segments = pred_segments[unmatched_indices].clone()

    pairwise_metric  = pairwise_segment_metric(matched_segments_merged, unmatched_segments, metric=metric)
    merge_indices = np.argmax(pairwise_metric, axis=0)

    for unmatch_idx, match_idx in enumerate(merge_indices):
        matched_segments_merged[match_idx] = torch.logical_or(unmatched_segments[unmatch_idx], matched_segments_merged[match_idx])
    
    return matched_segments_merged, merge_indices, unmatched_indices


def mask_to_rle_pytorch(tensor: torch.Tensor) -> List[Dict[str, Any]]:
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [
                torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device),
                cur_idxs + 1,
                torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device),
            ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})
    return out


def rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Compute a binary mask from an uncompressed RLE."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx = 0
    parity = False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    mask = mask.reshape(w, h)
    return mask.transpose()  # Put in C order


def coco_encode_rle(uncompressed_rle: Dict[str, Any]) -> Dict[str, Any]:
    from pycocotools import mask as mask_utils  # type: ignore

    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    return rle