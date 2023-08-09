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

from pytorch3d.renderer import CamerasBase, PerspectiveCameras

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


def get_axes_traces(tform=torch.eye(4), scale=0.3):

    axes = torch.cat([torch.zeros(3,1), torch.tensor([[1,0,0],
                                                      [0,1,0],
                                                      [0,0,1]])], dim=1)
    axes*=scale
    axes = torch.cat([axes, torch.ones(1,4)], dim=0)
    axes = torch.matmul(axes.transpose(0,1), tform)

    axis_start = axes[0]
    axis_x, axis_y, axis_z = axes[1], axes[2], axes[3]
    
    traces = []
    for axis_end,axis_color in zip([axis_x, axis_y, axis_z],
                                   ["red","green","blue"]):
        cone = go.Cone(x=[axis_end[0]],
                      y=[axis_end[1]],
                      z=[axis_end[2]],
                      u=[0.3*(axis_end[0]-axis_start[0])],
                      v=[0.3*(axis_end[1]-axis_start[1])],
                      w=[0.3*(axis_end[2]-axis_start[2])],
                      colorscale=[[0,axis_color],[1,axis_color]],
                      showscale=False)
        traces.append(cone)

        line = go.Scatter3d(x=[axis_start[0],axis_end[0]],
                         y=[axis_start[1],axis_end[1]],
                         z=[axis_start[2],axis_end[2]],
                         mode="lines",
                        line=dict(color=axis_color, width=4),
                         )
        traces.append(line)
        
    return traces

def get_cameras(
    intrinsics: torch.Tensor,
    extrinsics: torch.Tensor,
    image_sizes: torch.Tensor
) -> Type[CamerasBase]:
    """
    Assumes input extrinsics follow pytorch3d matrix format
    M = [
        [Rxx, Ryx, Rzx, 0],
        [Rxy, Ryy, Rzy, 0],
        [Rxz, Ryz, Rzz, 0],
        [Tx,  Ty,  Tz,  1],
    ]
    """
    assert intrinsics.shape[0]==extrinsics.shape[0]==image_sizes.shape[0]
    assert intrinsics.shape[1:]==(4,4)
    assert extrinsics.shape[1:]==(4,4)
    assert image_sizes.shape[1:]==(2,)
    
    return PerspectiveCameras(in_ndc=False, K=intrinsics, R=extrinsics[:,:3,:3], T=extrinsics[:,3,:3], image_size=image_sizes)

def get_xy_depth(
    depth_map: torch.Tensor, # B x 1 x H x W
    from_ndc: bool = False
) -> torch.Tensor:
    # assert depth_map.shape[2:]==image_size
    batch, _, height, width = depth_map.shape
    # print(batch,_,height,width)
    if from_ndc:
        h_start, h_end  = max(1, height/width), -max(1, height/width)
        w_start, w_end  = max(1, width/height), -max(1, width/height)
        h_step, w_step = -(h_start-h_end)/height, -(w_start-w_end)/width
    else:
        h_start, h_end = 0, height
        w_start, w_end = 0, width
        h_step, w_step = 1, 1
    grid_h, grid_w = torch.meshgrid(torch.arange(h_start, h_end, h_step), 
                                    torch.arange(w_start, w_end, w_step), indexing='ij')
    
    xy_map = torch.stack((grid_w, grid_h), dim=-1)
    # plt.imshow(xy_map[:,:,0])
    # plt.show()
    # plt.imshow(xy_map[:,:,1])
    # plt.show()
    xy_map = torch.tile(xy_map, (batch,1,1,1)).to(depth_map.device)
    xy_depth = torch.cat([xy_map, depth_map.permute(0,2,3,1)], axis=3)
    
    return xy_depth


def get_RT_inverse(
    R: torch.Tensor,
    T: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    helper to derive inverse R,T transform from rotation (Nx3x3) and translation (Nx3x1)
    """
    assert T.shape[0]==R.shape[0], "Sequence length broken"
    assert T.shape[1:]==(3,1), "Translation transform has invalid shape"
    assert R.shape[1:]==(3,3), "Rotation transform has invalid shape"
    
    R_ = R.transpose(1,2)
    T_ = -torch.matmul(R_, T)
    
    return R_, T_


def get_homogeneous_matrix(
    R: torch.Tensor,
    T: torch.Tensor
) -> torch.Tensor:
    """
    helper to derive Nx4x4 homogeneous transform from rotation (Nx3x3) and translation (Nx3x1)
    Output matrix follows pytorch3d matrix format
    NOTE: the rotation component is transposed
    M = [
                [Rxx, Ryx, Rzx, 0],
                [Rxy, Ryy, Rzy, 0],
                [Rxz, Ryz, Rzz, 0],
                [Tx,  Ty,  Tz,  1],
            ]
    """
    assert T.shape[0]==R.shape[0], "Sequence length broken"
    assert T.shape[1:]==(3,1), "Translation transform has invalid shape"
    assert R.shape[1:]==(3,3), "Rotation transform has invalid shape"

    R_ = R.transpose(1,2)
    T_ = T.reshape(-1,1,3)

    M = torch.cat((R_, T_), dim=1) # N,4,3

    # Convert matrix into Nx4x4 homogeneous form
    M = torch.nn.functional.pad(M, (0,1,0,0,0,0), mode='constant', value=0)
    
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

        self.MVPcam2pytorch3dviewcam = torch.tensor([[[-1.0,  0.0,  0.0,  0.0],
                                                     [ 0.0,  1.0,  0.0,  0.0],
                                                     [ 0.0,  0.0, -1.0,  0.0],
                                                     [ 0.0,  0.0,  0.0,  1.0]]],
                                                     dtype=torch.float32)
        self.pytorch3dviewcam2MVPcam = self.MVPcam2pytorch3dviewcam


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

        R_C2W = tforms.quaternion_to_matrix(rotation_sequence_tensor)
        T_C2W = position_sequence_tensor.reshape(-1,3,1)
        C2W = torch.matmul(self.pytorch3dviewcam2MVPcam, get_homogeneous_matrix(R_C2W, T_C2W)) # Note the order of mult. is flipped due to transposed M
        
        R_W2C, T_W2C = get_RT_inverse(R_C2W, T_C2W)
        W2C = torch.matmul(get_homogeneous_matrix(R_W2C, T_W2C), self.MVPcam2pytorch3dviewcam) # Note the order of mult. is flipped due to transposed M
        
        return C2W, W2C 
        

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
        C2W, W2C = self.get_pose_matrices(rotation_sequence, position_sequence)
        
        sample['camera'] = {
            'K': np.tile(np.array([[[fx,  0.0, cx,  0.0],
                                    [0.0, fy,  cy,  0.0],
                                    [0.0, 0.0, 0.0, 1.0], 
                                    [0.0, 0.0, 1.0, 0.0]]],
                                    dtype=np.float32),
                              (self.window_size,1,1)),
            # 'K_': np.tile(np.array([[[1/fx,  0.0,  -cx/fx, 0.0],
            #                          [0.0,   1/fy, -cy/fy, 0.0],
            #                          [0.0,   0.0,  0.0,    1.0], 
            #                          [0.0,   0.0,  1.0,    0.0]]],
            #                         dtype=np.float32),
            #                   (self.window_size,1,1)),
            'W2V_pose': np.array(W2C),
            'V2W_pose': np.array(C2W)
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
            'class_dict': class_dict,
            'image_size': np.array([img.shape[:2] for img in image_sequence])
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