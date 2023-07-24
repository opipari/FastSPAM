import cv2
import numpy as np

import torch
import scipy

from PIL import Image

from panopticapi.utils import rgb2id


def read_panomaskRGB(path):
    label = Image.open(path)
    label = np.array(label, dtype=np.uint8)
    label = rgb2id(label)
    return label


def mask_to_boundary(mask, dilation_ratio=0.02):
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


def segment_metrics(true_segment, pred_segment):
    """Calculate pairwise pixel-wise metrics (precision, recall, f-score, iou) between two sets of segments.

    Keyword arguments:
    true_segment -- binary numpy array of shape (H,W)
    pred_segment -- binary numpy array of shape (H,W)
    eps -- float to avoid numerical overflow for division
    """
    assert true_segment.dtype == pred_segment.dtype == bool
    assert true_segment.shape == pred_segment.shape

    intersection = np.sum((true_segment * pred_segment) > 0)
    precision_denominator = np.sum(pred_segment>0)
    recall_denominator = np.sum(true_segment>0)


    precision = intersection / precision_denominator
    recall = intersection / recall_denominator
    F_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    IOU_score = intersection / (precision_denominator + recall_denominator - intersection) if (precision_denominator + recall_denominator - intersection) > 0 else 0


    results = {"Precision": precision,
                "Recall": recall,
                "F-Score": F_score,
                "IOU": IOU_score,
                "Intersection": intersection,
                "Precision-Denominator": precision_denominator,
                "Recall-Denominator": recall_denominator
                }
    return results


