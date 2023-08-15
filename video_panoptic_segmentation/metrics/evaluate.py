import os
import json
import argparse


import numpy as np

from . import utils as metric_utils

from tqdm import tqdm



def evaluate_metrics(anno_dir_seqs, pred_dir_seqs, dataset):
    anno_dir, anno_sequences = anno_dir_seqs
    pred_dir, pred_sequences = pred_dir_seqs

    assert len(anno_sequences)==len(pred_sequences)
    assert set(anno_sequences)==set(pred_sequences)

    results = {}

    for sequence in tqdm(anno_sequences):
        annotations = next(v for v in dataset["annotations"] if v["video_name"]==sequence)["annotations"]

        results[sequence] = []

        for anno in annotations:

            anno_result = {}

            file_path = os.path.join(sequence, anno["file_name"])

            anno_file_path = os.path.join(anno_dir, file_path)
            pred_file_path = os.path.join(pred_dir, file_path)

            if not (os.path.isfile(anno_file_path) and os.path.isfile(pred_file_path)):
                continue

            anno_arr = metric_utils.read_panomaskRGB(anno_file_path)
            pred_arr = metric_utils.read_panomaskRGB(pred_file_path)

            assert set(np.unique(anno_arr)).difference(set([segment["id"] for segment in anno["segments_info"]]))=={0}
            

            for segment_id in np.sort(np.unique(anno_arr)):
                if segment_id!=0:
                    segment = next(seg for seg in anno["segments_info"] if seg["id"]==segment_id)
                    category_id = segment["category_id"]
                else:
                    category_id = 0


                anno_mask = anno_arr==segment_id
                pred_mask = pred_arr==segment_id

                
                
                anno_boundary = metric_utils.mask_to_boundary(anno_mask.astype(np.uint8), dilation_ratio=0.002)
                pred_boundary = metric_utils.mask_to_boundary(pred_mask.astype(np.uint8), dilation_ratio=0.002)
                anno_boundary = anno_boundary > 0
                pred_boundary = pred_boundary > 0


                mask_metrics = {"Mask-"+key: value for key,value in metric_utils.segment_metrics(anno_mask, pred_mask).items()}
                boundary_metrics = {"Boundary-"+key: value for key,value in metric_utils.segment_metrics(anno_boundary, pred_boundary).items()}

                anno_result[int(segment_id)] = {**mask_metrics, **boundary_metrics}
                anno_result[int(segment_id)]["category_id"] = int(category_id)

            results[sequence].append(anno_result)


    return results



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str, default='./VIPOSeg/valid')
    parser.add_argument('--pred_path',type=str, required=True)
    args = parser.parse_args()
    
    anno_file = os.path.join(args.data_path, 
        next(fl for fl in os.listdir(args.data_path) if fl.endswith('.json')))
    
    with open(anno_file, 'r') as fl:
        dataset = json.load(fl)

    anno_sequences = sorted([v["video_name"] for v in dataset["videos"]])
    assert all(os.path.isdir(os.path.join(args.data_path,"panomasksRGB",fldr)) for fldr in anno_sequences)

    pred_sequences = [fldr for fldr in os.listdir(os.path.join(args.pred_path,"panomasksRGB")) if os.path.isdir(os.path.join(args.pred_path, "panomasksRGB", fldr))]
    assert len(anno_sequences)==len(pred_sequences)
    assert set(anno_sequences)==set(pred_sequences)

    result_dict = evaluate_metrics((os.path.join(args.data_path,"panomasksRGB"), anno_sequences),
                                    (os.path.join(args.pred_path,"panomasksRGB"), pred_sequences),
                                    dataset)

    with open(os.path.join(args.pred_path, "metrics.json"),"w") as fl:
        json.dump(result_dict, fl)