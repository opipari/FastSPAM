import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image

from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

class VaryingPerspectiveDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform

        self.prefixes = sorted([fl.strip(".color.jpg") for fl in os.listdir(self.root_dir) if fl.endswith('color.jpg')], key=lambda x: int(x.split('.')[3]))
        self.len = len(self.prefixes)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]
        img_path = os.path.join(self.root_dir, prefix+'.color.jpg')
        label_path = os.path.join(self.root_dir, prefix+'.semantic.npy')

        image = read_image(img_path)
        label = np.load(label_path).astype(np.int32)
        label = np.stack([label==segment_id for segment_id in np.unique(label)], axis=0)
        label = torch.from_numpy(label)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label


    def convert_semantic_to_viz(self, label):
        label = label.numpy()
        label = np.sum(label*np.arange(len(label)).reshape(-1,1,1), axis=0, keepdims=False)
        semantic_img = Image.new("P", (label.shape[1], label.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((label.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")

        # Refernce code to approximately invert the color palette
        # label = np.where(np.all((np.expand_dims(label,2)==d3_40_colors_rgb), axis=3))[2].reshape(label.shape[:2])

        return semantic_img