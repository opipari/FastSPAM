import os
import csv


import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision


class Scene:
    def __init__(self, root, name):
        self.root = root
        self.name = name

        self.meta_filepath_all_view_poses = os.path.join(self.root, f"{self.name}.all_view_poses.csv")
        self.meta_filepath_acccepted_view_poses = os.path.join(self.root, f"{self.name}.accepted_view_poses.csv")
        self.meta_filepath_semantic = os.path.join(self.root, f"{self.name}.semantic.csv")

        self.views = self.__get_accepted_view_meta__()
        self.objects = self.__get_object_to_view_mapping__()

    def __len__(self):
        return len(self.views.keys())

    def __getitem__(self, view_idx):
        return self.views[view_idx]

    def __get_accepted_view_meta__(self):
        view_meta = {}
        
        with open(self.meta_filepath_acccepted_view_poses, 'r') as csvfile:

            pose_reader = csv.reader(csvfile, delimiter=',')

            for pose_meta in pose_reader:
                scene_name, view_idx, valid_view_idx, pos_idx, rot_idx, x_pos, y_pos, z_pos, roll, pitch, yaw = pose_meta
                
                # Skip information line if it is first
                if scene_name=='Scene-ID':
                    continue

                # Parse pose infomration out of string type
                view_idx, valid_view_idx, pos_idx, rot_idx = int(view_idx), int(valid_view_idx), int(pos_idx), int(rot_idx)
                x_pos, y_pos, z_pos = float(x_pos), float(y_pos), float(z_pos)
                roll, pitch, yaw = float(roll), float(pitch), float(yaw)

                view_meta[valid_view_idx] = {
                                            "prefix": f"{self.name}.{valid_view_idx:010}.{pos_idx:010}.{rot_idx:010}",
                                            "view_idx": view_idx,
                                            "valid_view_idx": valid_view_idx,
                                            "position_idx": pos_idx,
                                            "rotation_idx": rot_idx,
                                            "position_X": x_pos,
                                            "position_Y": y_pos,
                                            "position_Z": z_pos,
                                            "rotation_roll": roll,
                                            "rotation_pitch": pitch,
                                            "rotation_yaw": yaw
                                            }
                
        return view_meta


    def __get_object_to_view_mapping__(self):
        object_to_views = {}

        with open(self.meta_filepath_semantic, 'r') as csvfile:

            semantic_reader = csv.reader(csvfile, delimiter=',')

            for sem_meta in semantic_reader:
  
                object_id, object_hex_color, object_name = sem_meta[:3]
                views_of_object = sem_meta[3:]

                # Skip information line if it is first
                if object_id=='Object-ID':
                    continue

                object_rgb_color = np.uint8([int(object_hex_color[i:i+2], 16) for i in (0,2,4)])
                
                object_to_views[object_hex_color] = {"object_id": object_id,
                                                    "object_hex_color": object_hex_color, 
                                                    "object_rgb_color": object_rgb_color,
                                                    "object_name": object_name.strip("\""),
                                                    "visible_views": [int(valid_view_idx) for valid_view_idx in views_of_object]
                                                    }
        return object_to_views

    def read_image(self, view_idx, illumination=0):
        view_prefix = self.views[view_idx]["prefix"]
        image_path = os.path.join(self.root, view_prefix+f'.RGB.{illumination:010}.png')
        
        image = Image.open(image_path).convert('L').convert('RGB')
        image =  torchvision.transforms.functional.pil_to_tensor(image)

        return image, view_prefix

    def read_label(self, view_idx, object_id=None):
        view_prefix = self.views[view_idx]["prefix"]
        label_path = os.path.join(self.root, view_prefix+'.SEM.pt')

        label = torch.load(label_path)
        
        label_mask = label["semantic_label_mask"]
        label_hex_colors = label["semantic_label_hex_colors"]


        return label_mask, label_hex_colors, view_prefix

    def label_mask_2_label_image(self, label_mask, label_hex_colors):

        label_rgb_colors = torch.ByteTensor([[int(object_hex_color[i:i+2], 16) for i in (0,2,4)] for object_hex_color in label_hex_colors], device=label_mask.device)
        label_image = torch.sum(label_rgb_colors.reshape(-1,3,1,1) * label_mask.unsqueeze(1), dim=0)

        return label_image

class SceneLevelSegmentationDataset(Dataset):
    def __init__(self, root, illumination=0, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.illumination = illumination

        self.scenes = []
        for scene_dir in os.listdir(self.root):
            self.scenes.append(Scene(os.path.join(self.root, scene_dir), scene_dir))

        self.scene_lengths = [len(scene) for scene in self.scenes]

    def __len__(self):
        return sum(self.scene_lengths)

    def __getitem__(self, idx, scene_idx=None):
        if scene_idx is None:
            scene_idx = np.argmax(idx < np.cumsum(self.scene_lengths))
            view_idx = idx - ([0]+list(np.cumsum(self.scene_lengths)))[scene_idx]
        else:
            view_idx = idx

        scene = self.scenes[scene_idx]
        
        image, view_prefix = scene.read_image(view_idx)
        label_mask, label_hex_colors, view_prefix = scene.read_label(view_idx)

        if self.transform:
            image = self.transform(image)

        # if self.target_transform:
        #     label = self.target_transform(label)
            
        return image, (label_mask, label_hex_colors), scene.name, view_prefix


