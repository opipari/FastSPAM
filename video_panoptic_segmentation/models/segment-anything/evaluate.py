import os
import json
import argparse

from tqdm import tqdm

import numpy as np
import torch

from MVPd.utils.MVPdataset import MVPDataset, MVPVideo, MVPdCategories, video_collate
from MVPd.utils.MVPdHelpers import get_xy_depth, get_RT_inverse, get_pytorch3d_matrix, get_cameras, label_to_one_hot
from visualize import masks_to_panomasks


from segment_anything.modeling import Sam
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor


def get_dataset(dataset_config):
    return MVPDataset(root=dataset_config['root'],
                            split=dataset_config['split'],
                            training=dataset_config['training'],
                            window_size = dataset_config['window_size'])
    
def get_model(model_config, device):
    model_type = model_config['model_type']
    sam_checkpoint = model_config['sam_checkpoint']
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam = sam.to(device=device)
    model = SamAutomaticMaskGenerator(sam)
    return model


def evaluation_process(index, nprocs, config, output_dir):
    dataset = get_dataset(config['dataset'])
    nvideos_per_proc = math.ceil(len(dataset)/nprocs)
    subset_indices = torch.arange(index*nvideos_per_proc, min(((index+1)*nvideos_per_proc), len(dataset)))
    MVPdatasubset = torch.utils.data.Subset(dataset, subset_indices)

    model = get_model(config['model'])
    
    
    with torch.no_grad():
        for video in tqdm(MVPdatasubset):
            results = []
            first_sample = next(iter(video))
            video_name = first_sample['meta']['video_name']
            out_dir = os.path.join(output_dir, config['experiment_name'], 'panomasksRGB', video_name)
            os.makedirs(out_dir, exist_ok=True)

            for index, sample in enumerate(video):
                # Load metadata
                video_name = sample['meta']['video_name']
                out_dir = os.path.join(output_dir, config['experiment_name'], 'panomasksRGB', video_name)
                out_file = sample['meta']['window_names'][0].split('.')[0]+'.png'

                if video_name not in results:
                    results[video_name] = []

                # Load label data
                label = sample['label']['mask']    
                label_one_hot, label_ids = label_to_one_hot(label)
                if label_ids[0]==0:
                    label_one_hot = label_one_hot[1:]
                    label_ids = label_ids[1:]
                label_one_hot = torch.as_tensor(label_one_hot, device=model.predictor.device, dtype=torch.bool).squeeze(1)
                label_ids = torch.as_tensor(label_ids, device=label_one_hot.device, dtype=torch.int)
                label_rgbs = torch.as_tensor(id2rgb(label_ids.cpu().numpy()), device=label_one_hot.device, dtype=torch.int)

                # Run SAM
                image = np.array(sample['observation']['image'][0]).astype(np.uint8) # 480 x 640 x 3
                pred = model.generate(image)
                pred_masks = np.stack([seg['segmentation'] for seg in pred])
                pred_masks = torch.as_tensor(pred_masks, device=label_one_hot.device, dtype=torch.bool)

                # Filter SAM's predictions to 'align' with labeled segments
                matched_label_ind, matched_pred_ind, _ = metric_utils.match_segments(label_one_hot, pred_masks)
                unmatched_label_ind = np.setdiff1d(np.arange(label_one_hot.shape[0]), matched_label_ind, assume_unique=True)
                pred_masks_merged, _, _ = metric_utils.merge_unmatched_segments(predicted_masks, matched_pred_ind)
                pred_rgbs = np.array(label_rgbs[matched_label_ind].cpu().numpy())

                # Account for any labeled masks that have no matched prediction
                matched_ids = label_ids[matched_label_ind]
                matched_labels = label_one_hot[matched_label_ind]
                unmatched_ids = label_ids[unmatched_label_ind]

                frame_result = {}
                # Calculate boundary and mask statistics in pairwise matched fashion
                for matched_id, anno_mask, pred_mask in zip(matched_ids, matched_labels, pred_masks_merged):
                    anno_boundary = metric_utils.mask_to_boundary(anno_mask.cpu().numpy().astype(np.uint8), dilation_ratio=0.002)
                    pred_boundary = metric_utils.mask_to_boundary(pred_mask.cpu().numpy().astype(np.uint8), dilation_ratio=0.002)
                    anno_boundary = torch.as_tensor(anno_boundary > 0, device=label_one_hot.device, dtype=torch.bool)
                    pred_boundary = torch.as_tensor(pred_boundary > 0, device=label_one_hot.device, dtype=torch.bool)
                    mask_metrics = {"Mask-"+key: value for key,value in metric_utils.segment_metrics(anno_mask, pred_mask).items()}
                    boundary_metrics = {"Boundary-"+key: value for key,value in metric_utils.segment_metrics(anno_boundary, pred_boundary).items()}
                    frame_result[int(matched_id)] = {**mask_metrics, **boundary_metrics}
                    frame_result[int(matched_id)]['category'] = sample['meta']['class_dict'][int(matched_id)]
                # For any unmatched ground truth masks, assign empty metrics
                for unmatched_id in unmatched_ids:
                    frame_result[int(unmatched_id)] = {key:0.0 for key in ["Mask-Precision", "Mask-Recall", "Mask-F-Score", "Mask-IOU", "Mask-Intersection", "Mask-Precision-Denominator", "Mask-Recall-Denominator",
                                                                          "Boundary-Precision", "Boundary-Recall", "Boundary-F-Score", "Boundary-IOU", "Boundary-Intersection", "Boundary-Precision-Denominator", "Boundary-Recall-Denominator"]} 
                    frame_result[int(unmatched_id)]['category'] = sample['meta']['class_dict'][int(unmatched_id)]

                results.append(frame_result)
                
                pred_panomask = Image.fromarray((masks_to_panomasks(pred_masks_merged.cpu().numpy(), pred_rgbs/255)*255).astype(np.uint8))
                pred_panomask.save(os.path.join(out_dir, out_file))

            json.dump(results, open(os.path.join(out_dir,'result.json'),'w'))
            print("Finished processing", video_name)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", dest="config_path")
    parser.add_argument("--output-path", dest="output_path")
    args = parser.parse_args()

    config = json.load(open(args.config_path, 'r'))

    num_gpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(evaluation_process, args=(num_gpus, config, output_path), nprocs=num_gpus)

    