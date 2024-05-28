import os
import json

from typing import List, Optional, Tuple, Callable, Union

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
from torchvision import transforms as TF

import pytorch3d.transforms as tforms

from MVPd.utils.MVPdHelpers import get_RT_inverse, get_pytorch3d_matrix, filter_idmask_area

from natsort import natsorted



__all__ = [
    'video_collate',
    'ScanNetVideo',
    'ScanNetDataset',
]





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


class ScanNetVideo(Dataset):
    def __init__(
        self, 
        image_root: str,
        depth_root: str,
        pose_root: str,
        label_root: str,
        video_meta: dict,
        window_size: int = 1,
        transform: Optional[Callable[[dict], dict]] = None,
        rgb: bool = True,
    ) -> None:

        self.image_root = image_root
        self.depth_root = depth_root
        self.pose_root = pose_root
        self.label_root = label_root
        self.video_meta = video_meta
        self.window_size = window_size
        self.transform = transform
        self.rgb = rgb


        # Note that MVPCameras were defined in blender, with -Z forward, +X right, and +Y up
        # Meanwhile, PyTorch3D defines its camera view coordinate system with +Z forward, -X right, and +Y up
        # Hence, to convert between these coordinate systems, we use a 180deg rotation about Y, given below:
        self.MVPCamera2PyTorch3dViewCoSys = torch.tensor([[[-1.0,  0.0,  0.0,  0.0],
                                                             [ 0.0,  -1.0,  0.0,  0.0],
                                                             [ 0.0,  0.0, 1.0,  0.0],
                                                             [ 0.0,  0.0,  0.0,  1.0]]],
                                                             dtype=torch.float32)
        self.PyTorch3dViewCoSys2MVPCamera = self.MVPCamera2PyTorch3dViewCoSys


    def __len__(self) -> int:
        return len(self.video_meta["images"]) - self.window_size + 1


    def get_image_depth_label(
        self,
        idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        file_name = self.video_meta["images"][idx]

        image = cv2.imread(os.path.join(self.image_root, file_name))
        image = torch.tensor(image, dtype=torch.uint8)
        image = TF.functional.resize(image.permute(2,0,1), (480,640)).permute(1,2,0)
        image = np.array(image, dtype=np.float32)

        if self.rgb:
            image = image[:, :, [2, 1, 0]]

        depth_file = os.path.join(self.depth_root, '.'.join(file_name.split('.')[:-1])+'.png')
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED)
        depth = np.array(depth, dtype=np.float32)/1000.0
        assert depth.shape==(480,640)

        label = Image.open(os.path.join(self.label_root, '.'.join(file_name.split('.')[:-1])+'.png'))
        label = np.array(label, dtype=np.uint8)
        label = TF.functional.resize(torch.tensor(label).unsqueeze(0), (480,640), 
            interpolation=TF.InterpolationMode.NEAREST_EXACT).squeeze(0)
        label = np.asarray(label, dtype=np.uint8)

        return image, depth, label

    def get_pose(
        self,
        idx: int
    ) -> np.ndarray:
        file_num = self.video_meta["images"][idx]
        file_name = os.path.join(self.pose_root, '.'.join(file_num.split('.')[:-1])+'.txt')

        with open(file_name, 'r') as f:
            pose = np.array([[float(n) for n in line.rstrip('\n').split()] for line in f], dtype=np.float32)

        return pose

    def get_pose_matrices(
        self,
        pose_sequence: List,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Helper to convert quaternion and translation to homogeneous pose transformation matrices
        '''
        pose_sequence_tensor = torch.tensor(pose_sequence)

        # Calculate rotation and translation from MVPCamera to World
        R_C2W = pose_sequence_tensor[:,:3,:3]
        T_C2W = pose_sequence_tensor[:,:3,3:]
        # Convert rotation and translation into pytorch3d matrix form (transposed homogeneous matrix)
        C2W = get_pytorch3d_matrix(R_C2W, T_C2W)
        # Convert transformation to be from PyTorch3d Camera View Coordinate System to World frame: https://pytorch3d.org/docs/cameras
        # NOTE: the C2W matrix is in transposed form; hence, the following matrix multiplication is in reversed order to compensate. 
        # I.E. if working out on paper, you'll expect: matmul(MVPCamera->World, PyTorch3dCamViewCoSys->MVPCamera),
        # but instead we do matmul(PyTorch3dCamViewCoSys->MVPCamera, MVPCamera->World.T)
        # This works since PyTorch3dCamViewCoSys->MVPCamera is its own inverse (transpose)
        V2W = torch.matmul(self.PyTorch3dViewCoSys2MVPCamera, C2W) # Note the order of mult. is flipped due to transposed M
        
        # Calculate rotation and translation from World to MVPCamera
        R_W2C, T_W2C = get_RT_inverse(R_C2W, T_C2W)
        # Convert rotation and translation into pytorch3d matrix form (transposed homogeneous matrix)
        W2C = get_pytorch3d_matrix(R_W2C, T_W2C)
        # Convert transformation to be from World to pytorch3d Camera View Coordinate System: https://pytorch3d.org/docs/cameras
        # NOTE: the W2C matrix is in transposed form; hence, the following matrix multiplication is in reversed order to compensate.
        # I.E. if working out on paper, you'll expect: matmul(MVPCamera->PyTorch3dCamViewCoSys, World->MVPCamera),
        # but instead we do matmul(World->World->MVPCamera.T, MVPCamera->PyTorch3dCamViewCoSys)
        # This works since MVPCamera->CamViewCoSys is its own inverse (transpose)
        W2V = torch.matmul(W2C, self.MVPCamera2PyTorch3dViewCoSys)
        
        return V2W, W2V 
    

    def __getitem__(
        self,
        idx: int
    ) -> dict:

        start_idx = idx
        end_idx = start_idx + self.window_size

        image_sequence = []
        depth_sequence = []
        label_sequence = []

        id_sequence = []

        pose_sequence = []

        for sub_idx in range(start_idx, end_idx):
            image, depth, label = self.get_image_depth_label(sub_idx)
            
            image_sequence.append(image)
            depth_sequence.append(depth)
            label_sequence.append(label)

            id_sequence.append(list(np.sort(np.unique(label))))

            pose_sequence.append(self.get_pose(sub_idx))
        
            

        sample = {}

        sample['observation'] = {
            'image': np.array(image_sequence),
            'depth': np.array(depth_sequence)
        }


        fx = 575.547668
        fy = 577.459778
        cx = 323.171967
        cy = 236.417465
        V2W, W2V = self.get_pose_matrices(pose_sequence)
        
        sample['camera'] = {
            'K': np.tile(np.array([[[fx,  0.0, cx,  0.0],
                                    [0.0, fy,  cy,  0.0],
                                    [0.0, 0.0, 0.0, 1.0], 
                                    [0.0, 0.0, 1.0, 0.0]]],
                                    dtype=np.float32),
                              (self.window_size,1,1)),
            'W2V_pose': np.array(W2V),
            'V2W_pose': np.array(V2W)
        }

        sample['label'] = {
            'mask': np.array(label_sequence),
            'id': list(id_sequence)
        }

        sample['meta'] = {
            'video_name': self.video_meta['video_name'],
            'window_idxs': np.array([sub_idx for sub_idx in range(start_idx, end_idx)]),
            'window_names': [self.video_meta['images'][sub_idx] for sub_idx in range(start_idx, end_idx)],
            'image_size': np.array([img.shape[:2] for img in image_sequence])
        }
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    


class ScanNetDataset(Dataset):
    def __init__(
        self,
        root: str = './datasets/ScanNet',
        split: str = 'scans_test',
        window_size: int = 1,
        transform: Optional[Callable[[dict], dict]] = None,
        rgb: bool = True,
    ) -> None:

        self.root = root
        self.split = split
        self.window_size = window_size
        self.transform = transform
        self.rgb = rgb

        self.image_root = 'color'
        self.depth_root = 'depth'
        self.pose_root = 'pose'
        self.label_root = 'instance-filt'

        
        video_names = natsorted(os.listdir(os.path.join(self.root, self.split)))
        self.videos = [{
                        'video_name': v_name,
                        'images': natsorted(os.listdir(os.path.join(self.root, self.split, v_name, self.image_root))),
                        } for v_name in video_names]


        for i in range(len(self.videos)):
            v_path = os.path.join(self.root, self.split, self.videos[i]['video_name'])
            vimages = os.listdir(os.path.join(v_path, self.image_root))
            vdepths = os.listdir(os.path.join(v_path, self.depth_root))
            vposes = os.listdir(os.path.join(v_path, self.pose_root))
            vlabels = os.listdir(os.path.join(v_path, self.label_root))

            assert len(self.videos[i]["images"])==len(vimages)==len(vdepths)==len(vposes)==len(vlabels)

        windows_per_video = [len(v['images'])-self.window_size+1 for v in self.videos]
        self.cumulative_windows_per_video = np.cumsum(windows_per_video)
        print(f"{len(self.videos)} videos in ScanNet:{self.root}/{self.split}")

    def __len__(self) -> int:
        if self.window_size>0:
            return self.cumulative_windows_per_video[-1]
        else:
            return len(self.videos)

    def __getitem__(
        self,
        idx: int
    ) -> Union[dict, ScanNetVideo]:
        

        if self.window_size>0:
            video_idx = np.argmax(idx<self.cumulative_windows_per_video)
            window_idx = idx if video_idx==0 else idx-self.cumulative_windows_per_video[video_idx-1]

            v_path = os.path.join(self.root, self.split, self.videos[video_idx]['video_name'])

            return ScanNetVideo(os.path.join(v_path, self.image_root), 
                            os.path.join(v_path, self.depth_root),
                            os.path.join(v_path, self.pose_root),
                            os.path.join(v_path, self.label_root),
                            self.videos[video_idx], 
                            window_size = self.window_size,
                            transform = self.transform,
                            rgb = self.rgb,
                            )[window_idx]

        else:
            v_path = os.path.join(self.root, self.split, self.videos[idx]['video_name'])
            return ScanNetVideo(os.path.join(v_path, self.image_root), 
                            os.path.join(v_path, self.depth_root),
                            os.path.join(v_path, self.pose_root),
                            os.path.join(v_path, self.label_root),
                            self.videos[idx], 
                            window_size = 1,
                            transform = self.transform,
                            rgb = self.rgb,
                            )