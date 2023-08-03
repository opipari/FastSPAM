import os
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from panopticapi.utils import rgb2id



def video_collate(batch):
    from torch.utils.data._utils.collate import default_collate # for torch < v1.13
    elem = batch[0]
    if isinstance(elem, dict):
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
            # elif k=='pose':
            #     batched_meta = {}
            #     for km in elem['pose'].keys():
            #         batched_meta[km] = default_collate([b['pose'][km] for b in batch])
            #     batched_sample['pose'] = batched_meta
            # else:
            #     batched_sample[k] = default_collate([b[k] for b in batch])
                
        return batched_sample
    else:
        return torch.utils.data.default_collate(batch)



class MVPVideo(Dataset):
    def __init__(self, image_root, 
                        depth_root, 
                        label_root,
                        video_meta, 
                        anno_meta,
                        window_size = 1,
                        transform=None,
                        rgb=True
                        ):

        self.image_root = image_root
        self.depth_root = depth_root
        self.label_root = label_root
        self.video_meta = video_meta
        self.anno_meta = anno_meta
        self.window_size = window_size
        self.transform = transform
        self.rgb = rgb


    def __len__(self):
        return len(self.video_meta["images"]) - self.window_size + 1


    def get_image_depth_label(self, idx):
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


    def get_id_to_class_dict(self, idx):
        return {ann["id"]: ann["category_id"] for ann in self.anno_meta["annotations"][idx]["segments_info"]}


    def __getitem__(self, idx):

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
            'image_sequence': np.array(image_sequence),
            'depth_sequence': np.array(depth_sequence)
        }

        sample['pose'] = {
            'position': np.array(position_sequence),
            'rotation': np.array(rotation_sequence)
        }

        sample['label'] = {
            'masks': np.array(label_sequence),
            'boxes': list(box_sequence),
            'ids': list(id_sequence),
            'class_dict': class_dict
        }

        sample['meta'] = {
            'video_name': self.video_meta['video_name'],
            'window_idxs': np.array([sub_idx for sub_idx in range(start_idx, end_idx)]),
            'window_names': [self.video_meta['images'][sub_idx]["file_name"] for sub_idx in range(start_idx, end_idx)]
        }
        
        if self.transform is not None:
            sample = self.transform(sample)

        return sample

    


class MVPDataset(Dataset):
    def __init__(self,
                 root='./datasets/MVPd',
                 split='train',
                 training=True,
                 window_size = 1,
                 transform = None,
                 rgb = True
                 ):

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
    
    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __len__(self):
        if self.training:
            return self.cumulative_windows_per_video[-1]
        else:
            return len(self.dataset["videos"])


    def __getitem__(self, idx):
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