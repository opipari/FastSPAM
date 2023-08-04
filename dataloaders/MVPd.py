import os
import json
from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple, Type, Callable, Union

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

import pytorch3d.transforms as tforms

from panopticapi.utils import rgb2id



def video_collate(
    batch: dict
) -> dict:
    elem = batch[0]
    batched_sample = {}
    for k in elem.keys():
        batched_k = {}
        for km in elem[k].keys():
            if isinstance(elem[k][km], np.ndarray):
                batched_k[km] = torch.tensor(np.stack([b[k][km] for b in batch], axis=1))
            elif isinstance(elem[k][km], list):
                batched_k[km] = list(zip(*[b[k][km] for b in batch]))
            elif isinstance(elem[k][km], dict):
                batched_k[km] = [b[k][km] for b in batch]
            else:
                batched_k[km] = default_collate([b[k][km] for b in batch])
        
        batched_sample[k] = batched_k

    return batched_sample





def get_homogeneous_matrix(
    R: torch.Tensor,
    T: torch.Tensor
) -> torch.Tensor:
    """
    helper to derive Nx4x4 homogeneous transform from rotation (Nx3x3) and translation (Nx3x1)
    """
    assert T.shape[0]==R.shape[0], "Sequence length broken"
    assert T.shape[1:]==(3,1), "Translation transform has invalid shape"
    assert R.shape[1:]==(3,3), "Rotation transform has invalid shape"

    M = torch.cat((R, T), dim=2) # N,3,4

    # Convert matrix into Nx4x4 homogeneous form
    M = torch.nn.functional.pad(M, (0,0,0,1,0,0), mode='constant', value=0)
    M[:,3,3] = 1

    return M
    

class MVPVideo(Dataset):
    def __init__(
        self, 
        image_root: str,
        depth_root: str,
        label_root: str,
        video_meta: dict,
        anno_meta: dict,
        window_size: int = 1,
        transform: Optional[Callable[[dict], dict]] =None,
        rgb: bool = True
    ) -> None:

        self.image_root = image_root
        self.depth_root = depth_root
        self.label_root = label_root
        self.video_meta = video_meta
        self.anno_meta = anno_meta
        self.window_size = window_size
        self.transform = transform
        self.rgb = rgb

        self.MVPcam2pytorch3d = torch.tensor([[[-1.0,  0.0,  0.0,  0.0],
                                             [ 0.0,  1.0,  0.0,  0.0],
                                             [ 0.0,  0.0, -1.0,  0.0],
                                             [ 0.0,  0.0,  0.0,  1.0]]],
                                             dtype=torch.float32)
        self.pytorch3d2MVPcam = self.MVPcam2pytorch3d


    def __len__(self) -> int:
        return len(self.video_meta["images"]) - self.window_size + 1


    def get_image_depth_label(
        self,
        idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        image_meta = self.video_meta["images"][idx]
        anno_meta = self.anno_meta["annotations"][idx]

        image = cv2.imread(
            os.path.join(self.image_root, self.video_meta["video_name"], image_meta["file_name"]))
        image = np.array(image, dtype=np.float32)

        if self.rgb:
            image = image[:, :, [2, 1, 0]]

        depth = cv2.imread(
            os.path.join(self.depth_root, self.video_meta["video_name"], image_meta["depth_file_name"]), cv2.IMREAD_UNCHANGED)
        depth = np.array(depth, dtype=np.float32)/1000.0

        label = Image.open(
            os.path.join(self.label_root, self.video_meta["video_name"], anno_meta["file_name"]))
        label = np.array(label, dtype=np.uint8)
        label = rgb2id(label)

        return image, depth, label


    def get_id_to_class_dict(
        self, 
        idx: int
    ) -> dict:
        return {ann["id"]: ann["category_id"] for ann in self.anno_meta["annotations"][idx]["segments_info"]}

    def get_pose_matrices(
        self,
        rotation_sequence: List,
        position_sequence: List
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Helper to convert quaternion and translation to homogeneous pose transformation matrices
        '''
        position_sequence_tensor = torch.tensor(position_sequence)
        rotation_sequence_tensor = torch.tensor(rotation_sequence)

        R_W2C = tforms.quaternion_to_matrix(rotation_sequence_tensor)
        T_W2C = position_sequence_tensor.reshape(-1,3,1)
        W2C = torch.matmul(self.MVPcam2pytorch3d, get_homogeneous_matrix(R_W2C, T_W2C))
        
        R_C2W = R_W2C.transpose(1,2)
        T_C2W = -torch.matmul(R_C2W, T_W2C)
        C2W = torch.matmul(get_homogeneous_matrix(R_C2W, T_C2W), self.pytorch3d2MVPcam)
        
        return W2C, C2W
        

    def __getitem__(
        self,
        idx: int
    ) -> dict:

        start_idx = idx
        end_idx = start_idx + self.window_size

        image_sequence = []
        depth_sequence = []
        label_sequence = []

        box_sequence = []
        id_sequence = []

        position_sequence = []
        rotation_sequence = []

        class_dict = {}
        for sub_idx in range(start_idx, end_idx):
            image, depth, label = self.get_image_depth_label(sub_idx)

            image_sequence.append(image)
            depth_sequence.append(depth)
            label_sequence.append(label)

            box_sequence.append([seg["bbox"] for seg in self.anno_meta["annotations"][sub_idx]["segments_info"]])
            id_sequence.append([seg["id"] for seg in self.anno_meta["annotations"][sub_idx]["segments_info"]])

            position_sequence.append(self.video_meta["images"][sub_idx]["camera_position"])
            rotation_sequence.append(self.video_meta["images"][sub_idx]["camera_rotation"])
        
            class_dict.update(self.get_id_to_class_dict(sub_idx))

        sample = {}

        sample['observation'] = {
            'image': np.array(image_sequence),
            'depth': np.array(depth_sequence)
        }


        fx = 465.6029
        fy = 465.6029
        cx = 320.00
        cy = 240.00
        W2C, C2W = self.get_pose_matrices(rotation_sequence, position_sequence)
        
        sample['camera'] = {
            'K': np.tile(np.array([[[fx,  0.0, cx],
                                        [0.0, fy,  cy],
                                        [0.0, 0.0, 1.0]]], 
                                        dtype=np.float32),
                              (self.window_size,1,1)),
            'K_': np.tile(np.array([[[1/fx,  0.0,  -cx/fx],
                                        [0.0,   1/fy, -cy/fy],
                                        [0.0,   0.0,  1.0]]], 
                                        dtype=np.float32),
                              (self.window_size,1,1)),
            'W2C_pose': np.array(W2C),
            'C2W_pose': np.array(C2W)
        }

        sample['label'] = {
            'mask': np.array(label_sequence),
            'box': list(box_sequence),
            'id': list(id_sequence)
        }

        sample['meta'] = {
            'video_name': self.video_meta['video_name'],
            'window_idxs': np.array([sub_idx for sub_idx in range(start_idx, end_idx)]),
            'window_names': [self.video_meta['images'][sub_idx]["file_name"] for sub_idx in range(start_idx, end_idx)],
            'class_dict': class_dict
        }
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    


class MVPDataset(Dataset):
    def __init__(
        self,
        root: str = './datasets/MVPd',
        split: str = 'train',
        training: bool = True,
        window_size: int = 1,
        transform: Optional[Callable[[dict], dict]] = None,
        rgb: bool = True
    ) -> None:

        self.root = root
        self.split = split
        self.training = training
        if not training:
            assert window_size==1, "Invalid setting for evaluation mode"
        self.window_size = window_size
        self.transform = transform
        self.rgb = rgb

        self.image_root = os.path.join(root, f'{self.split}/imagesRGB')
        self.depth_root = os.path.join(root, f'{self.split}/imagesDEPTH')
        self.label_root = os.path.join(root, f'{self.split}/panomasksRGB')

        annotation_file = os.path.join(root, f'{self.split}/panoptic_{self.split}.json')
        self.dataset = json.load(open(annotation_file, 'r'))

        windows_per_video = [len(v["images"])-self.window_size+1 for v in self.dataset["videos"]]
        self.cumulative_windows_per_video = np.cumsum(windows_per_video)
        print(f"{len(self.dataset['videos'])} videos in MVPd:{self.root}/{self.split}")
    
    def train(self) -> None:
        self.training = True

    def eval(self) -> None:
        self.training = False

    def __len__(self) -> int:
        if self.training:
            return self.cumulative_windows_per_video[-1]
        else:
            return len(self.dataset["videos"])


    def __getitem__(
        self,
        idx: int
    ) -> Union[dict, MVPVideo]:
        if self.training:
            video_idx = np.argmax(idx<self.cumulative_windows_per_video)
            window_idx = idx if video_idx==0 else idx-self.cumulative_windows_per_video[video_idx-1]
            return MVPVideo(self.image_root, 
                            self.depth_root, 
                            self.label_root,
                            self.dataset["videos"][video_idx], 
                            self.dataset["annotations"][video_idx],
                            window_size = self.window_size,
                            transform = self.transform,
                            rgb = self.rgb)[window_idx]
        else:
            return MVPVideo(self.image_root, 
                            self.depth_root, 
                            self.label_root,
                            self.dataset["videos"][idx], 
                            self.dataset["annotations"][idx],
                            window_size = self.window_size,
                            transform = self.transform,
                            rgb = self.rgb
                            )