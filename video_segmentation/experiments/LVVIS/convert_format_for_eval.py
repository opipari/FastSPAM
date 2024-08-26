import os
import json
import torch
import numpy as np


in_data_json_file = '/home/ANT.AMAZON.COM/topipari/Downloads/MVPd_test_coco_1.json'
in_data_json_folder = '/home/ANT.AMAZON.COM/topipari/Downloads/ov2seg_resnet50_part1_results'
out_pt_folder = '/home/ANT.AMAZON.COM/topipari/Downloads/ov2seg_resnet50'


mvpd_data = json.load(open(in_data_json_file,'r'))
id_video_map = {mvpd_data['videos'][i]['id']: mvpd_data['videos'][i]['file_names'][0].split('/')[3] for i in range(len(mvpd_data['videos']))}

assert len(mvpd_data['videos'])==len(os.listdir(in_data_json_folder))

for vid_fl in os.listdir(in_data_json_folder):
	vid_id = int(vid_fl.split('_')[1].split('.')[0])
	vid_json_data = json.load(open(os.path.join(in_data_json_folder, vid_fl), 'r'))
	assert len(vid_json_data)==100
	vid_length = len(vid_json_data[0]['segmentations'])

	out_boxes = np.zeros((len(vid_json_data), 6))
	for seg_id in range(len(vid_json_data)):
		out_boxes[seg_id,0] = seg_id+1
		out_boxes[seg_id,5] = vid_json_data[seg_id]['score']

	for img_id in range(vid_length):
		
		out_segs = []
		for seg_id in range(len(vid_json_data)):
			out_segs.append(vid_json_data[seg_id]['segmentations'][img_id])
		out_dict = {'bbox_results': out_boxes, 'coco_rle': out_segs}
		out_fl = f'{img_id:010}.pt'
		out_folder = os.path.join(out_pt_folder, 'panomasksRLE', id_video_map[vid_id])
		os.makedirs(out_folder, exist_ok=True)
		out_path = os.path.join(out_folder, out_fl)
		assert not os.path.isfile(out_path)

		torch.save(out_dict, out_path)