import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision

from PIL import Image

def parse_semantic_txt(text_path='/media/topipari/0CD418EB76995EEF/SegmentationProject/zeroshot_rgbd/datasets/matterport/HM3D/example/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.semantic.txt'):
    import matplotlib.colors as colors
    with open(text_path, 'r') as f:
        data = f.readlines()
    hex_colors = [line.split(',')[1] for line in data[1:]]
    rgb_colors = np.array([[int(255*x) for x in colors.hex2color('#'+c)] for c in hex_colors])
    return rgb_colors


class ActiveIlluminationDataset(Dataset):
    def __init__(self, root_dir, illumination=None, colorspace='RGB', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.target_transform = target_transform
        self.illumination = illumination
        self.colorspace = colorspace

        self.prefixes = sorted([fl.strip(".SEM.png").split('.') for fl in os.listdir(self.root_dir) if fl.endswith('.SEM.png')], key=lambda x: int(x[1]))
        self.prefixes = self.prefixes[:1000]
        self.len = len(self.prefixes)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        prefix = self.prefixes[idx]
        img_path = os.path.join(self.root_dir, '.'.join(prefix+[str(self.illumination)])+'.RGB.png')
        label_path = os.path.join(self.root_dir, '.'.join(prefix)+'.SEM.png')

        image = Image.open(img_path).convert('L').convert('RGB')
        image =  torchvision.transforms.functional.pil_to_tensor(image)

        # label = read_image(label_path)
        # label = np.array(torch.permute(label,(1,2,0)))[:,:,:3]
        # label = np.where(np.all((np.expand_dims(label,2)==parse_semantic_txt()), axis=3))[2].reshape(label.shape[:2])
        # print(label.shape)
        # label = np.load(label_path).astype(np.int32)
        # label = np.stack([label==segment_id for segment_id in np.unique(label)], axis=0)
        # label = torch.from_numpy(label)

        if self.transform:
            image = self.transform(image)

        # if self.target_transform:
        #     label = self.target_transform(label)
            
        return image, None


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