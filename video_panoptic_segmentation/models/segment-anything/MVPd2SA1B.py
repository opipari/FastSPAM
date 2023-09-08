from typing import List, Optional, Tuple, Callable, Union

import numpy as np

import torch
from torch.utils.data import Dataset

import torchvision as tv
from torchvision import datapoints

import torchvision.transforms.v2.functional as F

from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from MVPd.utils.MVPdHelpers import get_xy_depth, get_RT_inverse, get_pytorch3d_matrix, get_cameras, label_to_one_hot



def sample_point_from_mask(bin_mask, samples=1):
    height, width = bin_mask.shape
    bin_mask = bin_mask.reshape(-1)

    if isinstance(bin_mask, np.ndarray):
        lib = np
        bin_mask = bin_mask.astype(bool)
    elif isinstance(bin_mask, torch.Tensor):
        lib = torch
        bin_mask = bin_mask.bool()
    else:
        raise NotImplementedError

    xv, yv = lib.meshgrid(lib.arange(width), lib.arange(height), indexing='xy')
    grid = lib.stack([xv,yv], axis=-1)
    grid_indices = grid.reshape(-1,2)[bin_mask]
    assert grid_indices.shape[0]>0, "Invalid binary mask, cannot be empty"

    shuffle_indices = np.random.choice(np.arange(grid_indices.shape[0]), size=samples, replace=True)
    
    grid_indices = grid_indices[shuffle_indices]
    return grid_indices


def bbox_from_mask(bin_mask):
    # Based on https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array
    rows = np.any(bin_mask, axis=1)
    cols = np.any(bin_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return np.array([cmin, rmin, cmax, rmax])




class SanitizeMasksPointsBoxes(torch.nn.Module):
    def __init__(self, min_size):
        super().__init__()
        self.min_size = min_size

    def forward(self, sample):
        masks = sample['masks']
        point_coords = sample['point_coords']
        point_labels = sample['point_labels']
        boxes = sample['boxes']
        to_keep_indices = []
        for i in range(len(masks)):
            if masks[i].sum()>self.min_size:
                to_keep_indices.append(i)

        sample['masks'] = masks[to_keep_indices]
        sample['point_coords'] = point_coords[to_keep_indices]
        sample['point_labels'] = point_labels[to_keep_indices]
        sample['boxes'] = boxes[to_keep_indices]

        return sample

class RecomputeBoxes(torch.nn.Module):
    def forward(self, sample):
        sample['boxes'] = datapoints.BoundingBox(tv.ops.masks_to_boxes(sample['masks']),
                                                    format=datapoints.BoundingBoxFormat.XYXY,
                                                    spatial_size=F.get_spatial_size(sample['image']))
        return sample

class ResamplePoints(torch.nn.Module):
    def forward(self, sample):

        # Sample point
        point_samples = []
        for i in range(len(sample['masks'])):
            point = sample_point_from_mask(sample['masks'][i])
            point = torch.cat([point, torch.zeros_like(point)], axis=1)
            point_samples.append(point)
        point_samples = torch.stack(point_samples)

        sample['point_coords'] = datapoints.BoundingBox(point_samples,
                                                    format=datapoints.BoundingBoxFormat.XYWH,
                                                    spatial_size=F.get_spatial_size(sample['image']))
        sample['point_labels'] = torch.ones(point_samples.shape[:2])
        
        return sample

class RandomSamplePointsAndBoxes(torch.nn.Module):
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def forward(self, sample):
        to_keep_indices = np.arange(len(sample['point_coords']))
        np.random.shuffle(to_keep_indices)
        to_keep_indices = to_keep_indices[:self.n_samples]

        sample['masks'] = sample['masks'][to_keep_indices]
        sample['point_coords'] = sample['point_coords'][to_keep_indices]
        sample['point_labels'] = sample['point_labels'][to_keep_indices]
        sample['boxes'] = sample['boxes'][to_keep_indices]

        return sample

class RandomDropPointsOrBoxes(torch.nn.Module):
    def __init__(self,p_points):
        super().__init__()
        self.p_points = p_points

    def forward(self, sample):
        if torch.rand(1).item()<self.p_points:
            del sample['point_coords']
            del sample['point_labels']
        else:
            del sample['boxes']
            sample['point_coords'] = sample['point_coords'][:,:,:2]
        return sample




class MVPd2SA1B(Dataset):
    def __init__(
        self,
        root: str = './datasets/MVPd',
        split: str = 'train',
        image_root: str = 'imagesRGB.0000000000',
        max_masks_per_sample: int = 64,
        transform: Optional[Callable[[dict], dict]] = None,
        rgb: bool = True
    ) -> None:

        self.root = root
        self.split = split
        self.max_masks_per_sample = max_masks_per_sample
        self.transform = transform
        self.rgb = rgb

        self.MVPd = MVPDataset(root=self.root,
                                split=self.split,
                                training=True,
                                window_size = 1)
    

    def __len__(self) -> int:
        return len(self.MVPd)


    def filter_masks(self, masks, min_=0.001):
        area = masks.shape[1]*masks.shape[2]
        to_keep_indices = []
        for i in range(len(masks)):
            if masks[i].sum()>(min_*area):
                to_keep_indices.append(i)
        return masks[to_keep_indices]


    def __getitem__(
        self,
        idx: int
    ) -> Union[dict, MVPVideo]:

        masks = []
        while len(masks)==0:
            MVPd_sample = self.MVPd[idx]

            image = MVPd_sample['observation']['image'][0].astype(np.uint8)
            image = np.transpose(image, axes=(2,0,1)) # C x H x W

            masks = MVPd_sample['label']['mask'][0]
            masks, ids = label_to_one_hot(masks, filter_void=True)
            masks = self.filter_masks(masks) # C x H x W
            if len(masks)==0:
                idx = np.random.randint(len(self))

        # Sample point
        point_samples = []
        for i in range(len(masks)):
            point = sample_point_from_mask(masks[i])
            point = np.concatenate([point, np.zeros_like(point)], axis=1)
            point_samples.append(point)
        point_samples = np.stack(point_samples)


        box_samples = []
        for i in range(len(masks)):
            box_samples.append(bbox_from_mask(masks[i]))
        box_samples = np.stack(box_samples)



        sample = {}
        sample['image'] = datapoints.Image(image)
        sample['masks'] = datapoints.Mask(masks)
        sample['original_size'] = (image.shape[1], image.shape[2])
        sample['boxes'] = datapoints.BoundingBox(box_samples,
                                                    format=datapoints.BoundingBoxFormat.XYXY,
                                                    spatial_size=F.get_spatial_size(sample['image']))
        sample['point_coords'] = datapoints.BoundingBox(point_samples,
                                                    format=datapoints.BoundingBoxFormat.XYWH,
                                                    spatial_size=F.get_spatial_size(sample['image']))
        sample['point_labels'] = torch.ones(point_samples.shape[:2])

        if self.transform:
            sample = self.transform(sample)

        return sample