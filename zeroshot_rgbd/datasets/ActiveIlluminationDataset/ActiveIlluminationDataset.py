import os
import csv

import scipy
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision


def parse_semantic_txt(text_path='/media/topipari/0CD418EB76995EEF/SegmentationProject/zeroshot_rgbd/datasets/matterport/HM3D/example/00861-GLAQ4DNUx5U/GLAQ4DNUx5U.semantic.txt'):
    import matplotlib.colors as colors
    with open(text_path, 'r') as f:
        data = f.readlines()
    hex_colors = [line.split(',')[1] for line in data[1:]]
    rgb_colors = np.array([[int(255*x) for x in colors.hex2color('#'+c)] for c in hex_colors])
    return rgb_colors


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
        label_path = os.path.join(self.root, view_prefix+'.SEM.png')

        label = Image.open(label_path).convert('RGB')
        label =  torchvision.transforms.functional.pil_to_tensor(label)

        label_rgb_colors = torch.unique(label.flatten(1,2).T, dim=0, sorted=True)
        if torch.all(label_rgb_colors[0,:]==0).item():
            label_rgb_colors = label_rgb_colors[1:]

        rgb2hex = lambda r,g,b: '%02X%02X%02X' % (r,g,b)
        label_hex_colors = [rgb2hex(*rgb_color) for rgb_color in label_rgb_colors]

        label_mask = torch.all(label_rgb_colors.reshape(-1,3,1,1) == label.unsqueeze(0), dim=1)

        return label, label_mask, label_hex_colors, view_prefix


class SceneDataset(Dataset):
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
        
        image, view_prefix = scene.read_image(view_idx, illumination=self.illumination)
        label, label_mask, label_hex_colors, view_prefix = scene.read_label(view_idx)

        if self.transform:
            image = self.transform(image)

        # if self.target_transform:
        #     label = self.target_transform(label)
            
        return image, (label, label_mask, label_hex_colors), scene.name, view_prefix


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


def get_view_segment_matches(predicted_segments, labeled_segments, eps=1e-10):
    """Calculate matched segments given a two segmentation candidates on a single view.

    Keyword arguments:
    predicted_segments -- binary tensor of shape (A,H,W) where A corresponds to number of predicted segments in a
    labeled_segments -- binary tensor of shape (B,H,W) where B corresponds to number of predicted segments in b
    eps -- float to avoid numerical overflow for division
    """

    num_segments_predicted, num_segments_labeled = predicted_segments.shape[0], labeled_segments.shape[0]
    mask_dim_predicted, mask_dim_labeled = predicted_segments.shape[1:], labeled_segments.shape[1:]
    
    assert predicted_segments.ndim == labeled_segments.ndim == 3
    assert predicted_segments.device == labeled_segments.device
    assert mask_dim_predicted == mask_dim_labeled

    pairwise_segment_intersection = torch.sum(predicted_segments.unsqueeze(1) * labeled_segments.unsqueeze(0), dim=(2,3)) + eps
    pairwise_segment_precision_denominator = torch.sum(predicted_segments, dim=(1,2)).unsqueeze(-1) # num_segments_predicted x 1
    pairwise_segment_recall_denominator = torch.sum(labeled_segments, dim=(1,2)).unsqueeze(0) # 1 x num_segments_predicted

    assert pairwise_segment_intersection.shape == (num_segments_predicted, num_segments_labeled)
    assert pairwise_segment_precision_denominator.shape == (num_segments_predicted, 1)
    assert pairwise_segment_recall_denominator.shape == (1, num_segments_labeled)

    pairwise_segment_precision = pairwise_segment_intersection / pairwise_segment_precision_denominator
    pairwise_segment_recall = pairwise_segment_intersection / pairwise_segment_recall_denominator
    pairwise_segment_F_score = (2 * pairwise_segment_precision * pairwise_segment_recall) / (pairwise_segment_precision + pairwise_segment_recall)

    predicted_ind, labeled_ind = scipy.optimize.linear_sum_assignment(np.array(pairwise_segment_F_score.cpu()), maximize=True)

    return predicted_ind, labeled_ind

def scene_level_metric(scene_dataset):
    for scene_idx, scene in enumerate(scene_dataset.scenes):
        scene_view_matches = {}

        for view_idx in scene.views.keys():

            view_label, view_label_mask, view_label_hex_colors = scene.read_label(view_idx)
            view_labeled_segments = view_label_mask
            view_predicted_segments = view_label_mask
            

            view_matches = get_view_segment_matches(view_predicted_segments, view_labeled_segments)

            scene_view_matches[view_idx] = view_matches

            # print(view_matches)

        for object_hex_color in scene.objects:
            for view_idx in scene.objects[object_hex_color]["visible_views"]:
                view_label, view_label_mask, view_label_hex_colors = scene.read_label(view_idx)
                view_labeled_segments = view_label_mask
                view_predicted_segments = view_label_mask
                

                view_matches = scene_view_matches[view_idx]
                predicted_match_ind, labeled_match_ind = view_matches
                predicted_match_ind, labeled_match_ind = list(predicted_match_ind), list(labeled_match_ind)

                # Select the labeled segment for currernt object
                object_idx_in_label_mask = view_label_hex_colors.index(object_hex_color)
                view_object_label_mask = view_labeled_segments[object_idx_in_label_mask]

                # Select the predicted segment that matched to this specific labeled segment
                # Find the index in labeled_match_ind corresponding to current object
                object_idx_in_view_matches = labeled_match_ind.index(object_idx_in_label_mask)
                # Use the above index to index into predicted_match_ind to find index of matched predicted segment from predicted mask output
                view_object_predicted_mask = view_predicted_segments[predicted_match_ind[object_idx_in_view_matches]]

                print(view_object_label_mask.shape, view_object_predicted_mask.shapes)

if __name__=="__main__":
    dset = SceneDataset(root="./zeroshot_rgbd/datasets/renders/example", illumination=0)
    scene_level_metric(dset)



# def scene_level_metric:

#     for scene in scenes:
#         scene_view_matches = []
#         for view in scene.views:
#             predicted_segments = gen_pred(view)
#             labeled_segments = get_label(view, all_objects)

#             view_matches = get_view_segment_matches(predicted_segments, labeled_segments)

#             scene_view_matches.append(view_matches)

#         for object in scene.objects:
#             intersection = 0
#             union = 0
#             for view in object.views:
#                 view_object_label = get_label(view, object)
#                 view_object_pred = view_matches[view][view_object_label]
#                 intersection += intersection(view_object_label, view_object_pred)
#                 union += union(view_object_label, view_object_pred)



# def metrics(pred_img, label_img, eps=1e-10):

#     num_gt = len(label_img)
#     num_pred = len(pred_img)
#     #print("num gt:", num_gt)
#     #print("num pred:", num_pred)

#     # confusion matrix (pred x gt)
#     intersection_mat = np.zeros((num_pred, num_gt))
#     precision_denom = np.sum(pred_img, axis=(1,2)).reshape(num_pred, 1)
#     recall_denom = np.sum(label_img, axis=(1,2)).reshape(1, num_gt)
#     #precision_mat = np.zeros((num_pred, num_gt))
#     #recall_mat = np.zeros((num_pred, num_gt))


#     for pred_idx in range(num_pred):
#         pred_mask = pred_img[pred_idx]

#         for gt_idx in range(num_gt):
#             gt_mask = label_img[gt_idx]
            
#             intersection_mat[pred_idx][gt_idx] = np.sum(gt_mask * pred_mask) + eps
#             #precision_mat[pred_idx][gt_idx] = intersection / np.sum(pred_mask)
#             #recall_mat[pred_idx][gt_idx] = intersection / np.sum(gt_mask)

#     precision_mat = intersection_mat / precision_denom
#     recall_mat = intersection_mat / recall_denom
#     F_mat = (2*precision_mat*recall_mat) / (precision_mat + recall_mat)

#     assignment = linear_sum_assignment(-F_mat)

#     intersection_total = 0
#     precision_denom_total = 0
#     recall_denom_total = 0
#     for pred_idx, gt_idx in zip(*assignment):
#         intersection_total += intersection_mat[pred_idx][gt_idx]
#         precision_denom_total += precision_denom[pred_idx][0]
#         recall_denom_total += recall_denom[0][gt_idx]

#     return intersection_total, precision_denom_total, recall_denom_total




# class ActiveIlluminationDataset(Dataset):
#     def __init__(self, root_dir, illumination=None, colorspace='RGB', transform=None, target_transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.target_transform = target_transform
#         self.illumination = illumination
#         self.colorspace = colorspace

#         self.prefixes = sorted([fl.strip(".SEM.png").split('.') for fl in os.listdir(self.root_dir) if fl.endswith('.SEM.png')], key=lambda x: int(x[1]))
#         self.prefixes = self.prefixes[:1000]
#         self.len = len(self.prefixes)

#     def __len__(self):
#         return self.len

#     def __getitem__(self, idx):
#         prefix = self.prefixes[idx]
#         img_path = os.path.join(self.root_dir, '.'.join(prefix+[str(self.illumination)])+'.RGB.png')
#         label_path = os.path.join(self.root_dir, '.'.join(prefix)+'.SEM.png')

#         image = Image.open(img_path).convert('L').convert('RGB')
#         image =  torchvision.transforms.functional.pil_to_tensor(image)

#         # label = read_image(label_path)
#         # label = np.array(torch.permute(label,(1,2,0)))[:,:,:3]
#         # label = np.where(np.all((np.expand_dims(label,2)==parse_semantic_txt()), axis=3))[2].reshape(label.shape[:2])
#         # print(label.shape)
#         # label = np.load(label_path).astype(np.int32)
#         # label = np.stack([label==segment_id for segment_id in np.unique(label)], axis=0)
#         # label = torch.from_numpy(label)

#         if self.transform:
#             image = self.transform(image)

#         # if self.target_transform:
#         #     label = self.target_transform(label)
            
#         return image, None


#     def convert_semantic_to_viz(self, label):
#         label = label.numpy()
#         label = np.sum(label*np.arange(len(label)).reshape(-1,1,1), axis=0, keepdims=False)
#         semantic_img = Image.new("P", (label.shape[1], label.shape[0]))
#         semantic_img.putpalette(d3_40_colors_rgb.flatten())
#         semantic_img.putdata((label.flatten() % 40).astype(np.uint8))
#         semantic_img = semantic_img.convert("RGBA")

#         # Refernce code to approximately invert the color palette
#         # label = np.where(np.all((np.expand_dims(label,2)==d3_40_colors_rgb), axis=3))[2].reshape(label.shape[:2])

#         return semantic_img