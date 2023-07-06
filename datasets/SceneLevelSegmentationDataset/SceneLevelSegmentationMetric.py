import os
import scipy
import numpy as np

import torch
from SceneLevelSegmentationDataset import SceneLevelSegmentationDataset




def get_pairwise_segment_metrics(segments_A, segments_B, eps=1e-10):
    """Calculate pairwise pixel-wise metrics (precision, recall, f-score, iou) between two sets of segments.

    Keyword arguments:
    segments_A -- binary tensor of shape (A,H,W) where A corresponds to number of predicted segments in a
    segments_B -- binary tensor of shape (B,H,W) where B corresponds to number of predicted segments in b
    eps -- float to avoid numerical overflow for division
    """

    num_segments_A, num_segments_B = segments_A.shape[0], segments_B.shape[0]
    mask_dim_A, mask_dim_B = segments_A.shape[1:], segments_B.shape[1:]
    
    assert segments_A.ndim == segments_B.ndim == 3
    assert segments_A.device == segments_B.device
    assert mask_dim_A == mask_dim_B

    # Vectorized intersection calculation is fastest but runs into memory issues with large number of segments
    # pairwise_segment_intersection = torch.sum(segments_A.unsqueeze(1) * segments_B.unsqueeze(0), dim=(2,3)) + eps
    
    pairwise_segment_intersection = torch.zeros((num_segments_A, num_segments_B), device=segments_A.device)
    for idx_a in range(num_segments_A):
        seg_a = segments_A[idx_a]

        # Single loop is inbetween vectorized and double loop for efficiency
        pairwise_segment_intersection[idx_a] = torch.sum(seg_a.unsqueeze(0) * segments_B, dim=(1,2))
        
        # Double loop is slower but more memory efficient        
        # for idx_b in range(num_segments_B):
        #     seg_b = segments_B[idx_b]
        #     pairwise_segment_intersection[idx_a][idx_b] = torch.sum(seg_a * seg_b) + eps

    pairwise_segment_precision_denominator = torch.sum(segments_A, dim=(1,2)).unsqueeze(-1) # num_segments_A x 1
    pairwise_segment_recall_denominator = torch.sum(segments_B, dim=(1,2)).unsqueeze(0) # 1 x num_segments_B

    assert pairwise_segment_intersection.shape == (num_segments_A, num_segments_B)
    assert pairwise_segment_precision_denominator.shape == (num_segments_A, 1)
    assert pairwise_segment_recall_denominator.shape == (1, num_segments_B)

    pairwise_segment_precision = pairwise_segment_intersection / pairwise_segment_precision_denominator
    pairwise_segment_recall = pairwise_segment_intersection / pairwise_segment_recall_denominator
    pairwise_segment_F_score = (2 * pairwise_segment_precision * pairwise_segment_recall) / (pairwise_segment_precision + pairwise_segment_recall + eps)
    pairwise_segment_IOU_score = pairwise_segment_intersection / (pairwise_segment_precision_denominator + pairwise_segment_recall_denominator - pairwise_segment_intersection + eps)


    results = {"Precision": pairwise_segment_precision,
                "Recall": pairwise_segment_recall,
                "F-Score": pairwise_segment_F_score,
                "IOU": pairwise_segment_IOU_score,
                "Intersection": pairwise_segment_intersection,
                "Precision-Denominator": pairwise_segment_precision_denominator,
                "Recall-Denominator": pairwise_segment_recall_denominator
                }
    return results




def get_view_segment_matches(predicted_segments, labeled_segments):
    """Calculate matched segments given a two segmentation candidates on a single view.

    Keyword arguments:
    predicted_segments -- binary tensor of shape (A,H,W) where A corresponds to number of predicted segments in a
    labeled_segments -- binary tensor of shape (B,H,W) where B corresponds to number of predicted segments in b
    eps -- float to avoid numerical overflow for division
    """

    pairwise_segment_F_score = get_pairwise_segment_metrics(predicted_segments, labeled_segments)["F-Score"]

    predicted_ind, labeled_ind = scipy.optimize.linear_sum_assignment(np.array(pairwise_segment_F_score.cpu()), maximize=True)

    # Account for the possibility that linear sum may assign predicted segments to true segments with no overlap
    # In this case, removew the assignment from consideration to allow for downstreat merge of prediction into better assignement
    no_overlap_inds = [idx for idx in range(len(predicted_ind)) if pairwise_segment_F_score[predicted_ind[idx]][labeled_ind[idx]]==0]

    predicted_ind, labeled_ind = np.delete(predicted_ind, no_overlap_inds), np.delete(labeled_ind, no_overlap_inds)

    return predicted_ind, labeled_ind, pairwise_segment_F_score


def merge_unmatched_segments(predicted_segments, matched_indices):

    unmatched_indices = np.setdiff1d(np.arange(predicted_segments.shape[0]), matched_indices, assume_unique=True)

    matched_indices_merged = np.arange(matched_indices.shape[0])
    matched_segments_merged = predicted_segments[matched_indices].clone().detach()
    unmatched_segments = predicted_segments[unmatched_indices].clone().detach()

    pairwise_segment_F_score  = get_pairwise_segment_metrics(unmatched_segments, matched_segments_merged)["F-Score"]
    F_score_matches = torch.argmax(pairwise_segment_F_score, dim=1)

    for unmatch_idx, match_idx in enumerate(F_score_matches):
        matched_segments_merged[match_idx] = torch.logical_or(unmatched_segments[unmatch_idx], matched_segments_merged[match_idx])
    
    
    return matched_segments_merged, matched_indices_merged

def ignore_invalid_label_pixles(view_labeled_segments, view_predicted_segments, view_label_hex_colors):
    if view_label_hex_colors[0]=='000000':

        valid_mask = torch.logical_not(view_labeled_segments[0])
        
        view_labeled_segments = view_labeled_segments[1:]
        view_label_hex_colors = view_label_hex_colors[1:]

        view_predicted_segments = valid_mask.unsqueeze(0) * view_predicted_segments
    
    return view_labeled_segments, view_predicted_segments, view_label_hex_colors


def ignore_empty_masks(view_labeled_segments, view_predicted_segments, view_label_hex_colors):
    nonempty_labeled_segment_indices = (torch.sum(view_labeled_segments, dim=(1,2))>0).nonzero().squeeze()

    if len(nonempty_labeled_segment_indices.size())>0:
        view_labeled_segments = view_labeled_segments[nonempty_labeled_segment_indices]
        view_label_hex_colors = [view_label_hex_colors[idx] for idx in nonempty_labeled_segment_indices]

    nonempty_predicted_segment_indices = (torch.sum(view_predicted_segments, dim=(1,2))>0).nonzero().squeeze()
    if len(nonempty_predicted_segment_indices.size())>0:
        view_predicted_segments = view_predicted_segments[nonempty_predicted_segment_indices]

    return view_labeled_segments, view_predicted_segments, view_label_hex_colors

def SceneLevelSegmentationMetric(scene_dataset, segmentation_model, visualize=True):
    results = {}

    for scene_idx, scene in enumerate(scene_dataset.scenes):
        
        assert scene.name not in results
        results[scene.name] = {}

        for object_hex_color in scene.objects:
            results[scene.name][object_hex_color] = {}

        for view_idx in scene.views.keys():

            # if view_idx%10==0 and view_idx>0:
            #     all_precisions = [results[scene_name][object_hex_color][vidx]["Precision"] for scene_name in results for object_hex_color in results[scene_name] for vidx in results[scene_name][object_hex_color]]
            #     all_recalls = [results[scene_name][object_hex_color][vidx]["Recall"] for scene_name in results for object_hex_color in results[scene_name] for vidx in results[scene_name][object_hex_color]]
            #     all_f_scores = [results[scene_name][object_hex_color][vidx]["F-score"] for scene_name in results for object_hex_color in results[scene_name] for vidx in results[scene_name][object_hex_color]]
            #     all_ious = [results[scene_name][object_hex_color][vidx]["IOU"] for scene_name in results for object_hex_color in results[scene_name] for vidx in results[scene_name][object_hex_color]]

            #     print("Average precision:", sum(all_precisions)/len(all_precisions))
            #     print("Average recall:", sum(all_recalls)/len(all_recalls))
            #     print("Average F-score:", sum(all_f_scores)/len(all_f_scores))
            #     print("Average IoU:", sum(all_ious)/len(all_ious))

            #     print(view_idx, len(scene.views.keys()))


            view_labeled_segments, view_labeled_hex_colors, view_prefix = scene.read_label(view_idx)            
            view_predicted_segments = segmentation_model.get_output(view_prefix)
            

            # Move label and prediction to cuda for faster computing
            if torch.cuda.is_available():
                view_labeled_segments = view_labeled_segments.cuda()
                view_predicted_segments = view_predicted_segments.cuda()


            # Account for invalid label pixels where rendered semantics hit empty mesh (pixel hex=='000000' in semantic label)
            view_labeled_segments, view_predicted_segments, view_labeled_hex_colors = ignore_invalid_label_pixles(view_labeled_segments, view_predicted_segments, view_labeled_hex_colors)

            # After removing invalid pixels, account for any masks that are completely empty and remove from label or prediction
            view_labeled_segments, view_predicted_segments, view_labeled_hex_colors = ignore_empty_masks(view_labeled_segments, view_predicted_segments, view_labeled_hex_colors)

            matched_predicted_indices, matched_labeled_indices, pairwise_segment_F_score = get_view_segment_matches(view_predicted_segments, view_labeled_segments)
            if visualize:
                import matplotlib.pyplot as plt
                for v0,v1 in zip(matched_predicted_indices, matched_labeled_indices):
                    print('Looking at prediction with matched F-score:', pairwise_segment_F_score[v0][v1], 'non-zer-F-score:', pairwise_segment_F_score[v0][v1]==0, 'color:', view_labeled_hex_colors[v1], 'idx-pred:', v0, 'idx-label:', v1)
                    plt.imshow(scene.label_mask_2_label_image(view_predicted_segments[v0].unsqueeze(0).cpu(), [view_labeled_hex_colors[v1]]).permute(1,2,0))
                    plt.show()
                    
                    plt.imshow(scene.label_mask_2_label_image(view_labeled_segments[v1].unsqueeze(0).cpu(), [view_labeled_hex_colors[v1]]).permute(1,2,0))
                    plt.show()
                    
                

            view_predicted_segments_merged, matched_predicted_indices = merge_unmatched_segments(view_predicted_segments, matched_predicted_indices)
            if visualize:
                plt.imshow(scene.label_mask_2_label_image(view_predicted_segments_merged.cpu(), [view_labeled_hex_colors[idx] for idx in view_matches[1]]).permute(1,2,0))
                plt.show()

            matched_view_labeled_hex_colors = [view_labeled_hex_colors[idx] for idx in range(len(view_labeled_hex_colors)) if idx in matched_labeled_indices]
            unmatched_view_labeled_hex_colors = [(view_labeled_hex_colors[idx], idx) for idx in range(len(view_labeled_hex_colors)) if idx not in matched_labeled_indices]
            assert len(matched_view_labeled_hex_colors)+len(unmatched_view_labeled_hex_colors)==len(view_labeled_hex_colors) and len(matched_view_labeled_hex_colors)==len(matched_labeled_indices)

            for (object_hex_color, matched_predicted_idx, matched_labeled_idx) in zip(matched_view_labeled_hex_colors, matched_predicted_indices, matched_labeled_indices):

                metrics = get_pairwise_segment_metrics(view_predicted_segments_merged[matched_predicted_idx].unsqueeze(0), view_labeled_segments[matched_labeled_idx].unsqueeze(0))

                results[scene.name][object_hex_color][view_idx] = {"Precision": metrics["Precision"].item(),
                                                                    "Recall": metrics["Recall"].item(),
                                                                    "F-Score": metrics["F-Score"].item(),
                                                                    "IOU": metrics["IOU"].item(),
                                                                    "Intersection": metrics["Intersection"].item(),
                                                                    "Precision-Denominator": metrics["Precision-Denominator"].item(),
                                                                    "Recall-Denominator": metrics["Recall-Denominator"].item()
                                                                    }

            for (object_hex_color, unmatched_labeled_idx) in unmatched_view_labeled_hex_colors:
                metrics = get_pairwise_segment_metrics(torch.zeros_like(view_labeled_segments[unmatched_labeled_idx]).unsqueeze(0), view_labeled_segments[unmatched_labeled_idx].unsqueeze(0))

                results[scene.name][object_hex_color][view_idx] = {"Precision": 0.0,
                                                                    "Recall": 0.0,
                                                                    "F-Score": 0.0,
                                                                    "IOU": 0.0,
                                                                    "Intersection": metrics["Intersection"],
                                                                    "Precision-Denominator": metrics["Precision-Denominator"],
                                                                    "Recall-Denominator": metrics["Recall-Denominator"]
                                                                    }

    all_precisions = [results[scene_name][object_hex_color][vidx]["Precision"] for scene_name in results for object_hex_color in results[scene_name] for vidx in results[scene_name][object_hex_color]]
    all_recalls = [results[scene_name][object_hex_color][vidx]["Recall"] for scene_name in results for object_hex_color in results[scene_name] for vidx in results[scene_name][object_hex_color]]
    all_f_scores = [results[scene_name][object_hex_color][vidx]["F-Score"] for scene_name in results for object_hex_color in results[scene_name] for vidx in results[scene_name][object_hex_color]]
    all_ious = [results[scene_name][object_hex_color][vidx]["IOU"] for scene_name in results for object_hex_color in results[scene_name] for vidx in results[scene_name][object_hex_color]]

    print("Average precision:", sum(all_precisions)/len(all_precisions))
    print("Average recall:", sum(all_recalls)/len(all_recalls))
    print("Average F-Score:", sum(all_f_scores)/len(all_f_scores))
    print("Average IoU:", sum(all_ious)/len(all_ious))

    torch.save(results, f"./{scene_dataset.illumination:010}_results.pt")
            



class SAM_Reader_Model:
    def __init__(self, root, illumination):
        self.root = root
        self.illumination = illumination

    def get_output(self, view_prefix):
        return torch.load(os.path.join(self.root,
                                        f"output.{self.illumination:010}",
                                        view_prefix.split('.')[0],
                                        view_prefix+f'.RGB.{self.illumination:010}.SAM.pt')
                        )


if __name__=="__main__":
    for illumination in [0,20,200]:
        dset = SceneLevelSegmentationDataset(root="/media/topipari/0CD418EB76995EEF/SegmentationProject/zeroshot_rgbd/datasets/renders/example", illumination=illumination)
        sam_model_standin = SAM_Reader_Model(root="./zeroshot_rgbd/models", illumination=illumination)
        SceneLevelSegmentationMetric(dset, sam_model_standin)
