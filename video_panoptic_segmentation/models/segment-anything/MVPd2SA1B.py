from typing import List, Optional, Tuple, Callable, Union

import numpy as np

import torch
from torch.utils.data import Dataset


from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from MVPd.utils.MVPdHelpers import get_xy_depth, get_RT_inverse, get_pytorch3d_matrix, get_cameras, label_to_one_hot


# def simulate_interactive_prompt():


def sample_point_from_mask(bin_mask, samples=1):
    height, width = bin_mask.shape
    xv, yv = np.meshgrid(np.arange(width), np.arange(height), indexing='xy')
    grid = np.stack([xv,yv], axis=-1)
    bin_mask = bin_mask.reshape(-1)
    grid_indices = grid.reshape(-1,2)[bin_mask.astype(bool)]
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


    def filter_masks(self, masks, max_=0.9):
        area = masks.shape[1]*masks.shape[2]
        to_keep_indices = []
        for i in range(len(masks)):
            if masks[i].sum()>0 and masks[i].sum()<(max_*area):
                to_keep_indices.append(i)
        return masks[to_keep_indices]


    def __getitem__(
        self,
        idx: int
    ) -> Union[dict, MVPVideo]:

        MVPd_sample = self.MVPd[idx]


        image = MVPd_sample['observation']['image'][0]
        masks = MVPd_sample['label']['mask'][0]

        masks, ids = label_to_one_hot(masks)
        
        masks = self.filter_masks(masks)

        # Sample point
        point_samples = []
        for i in range(len(masks)):
            point_samples.append(sample_point_from_mask(masks[i]))
        point_samples = np.stack(point_samples)


        box_samples = []
        for i in range(len(masks)):
            box_samples.append(bbox_from_mask(masks[i]))
        box_samples = np.stack(box_samples)

        sample = {}
        sample['image'] = image
        sample['masks'] = masks
        sample['bbox'] = box_samples
        sample['points'] = point_samples

        if self.transform:
            sample = self.transform(sample)

        return sample