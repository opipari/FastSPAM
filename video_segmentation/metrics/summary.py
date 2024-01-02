import os
import json
# results = json.load(open('video_panoptic_segmentation/models/segment-anything/results/evaluate_pretrained_sam_automatic/metrics_sam_automatic.json','r'))
# results = json.load(open('video_panoptic_segmentation/models/segment-anything/results/evaluate_pretrained_sam_automatic_vit_b/metrics_sam_automatic_vit_b.json','r'))


out_path = "./video_panoptic_segmentation/models/segment-anything/results/evaluate_pretrained_sam_automatic/metricsJSON"


mask_inter = 0
mask_prec_den = 0
mask_rec_den = 0

bound_inter = 0
bound_prec_den = 0
bound_rec_den = 0

tot_videos = 0
tot_images = 0
tot_segments = 0

for video_name in os.listdir(out_path):
	
	tot_videos += 1

	metric_path = os.path.join(out_path, video_name, "metrics.json")
	if os.path.exists(metric_path):
		sample_results = json.load(open(metric_path,'r'))

		for ind in sample_results:
			for seg in sample_results[ind]:
				mask_inter_ = sample_results[ind][seg]["Mask-Intersection"]
				mask_prec_ = sample_results[ind][seg]["Mask-Precision-Denominator"]
				mask_rec_ = sample_results[ind][seg]["Mask-Recall-Denominator"]
				bound_inter_ = sample_results[ind][seg]["Boundary-Intersection"]
				bound_prec_ = sample_results[ind][seg]["Boundary-Precision-Denominator"]
				bound_rec_ = sample_results[ind][seg]["Boundary-Recall-Denominator"]


				mask_inter += mask_inter_
				mask_prec_den += mask_prec_
				mask_rec_den += mask_rec_

				bound_inter += bound_inter_
				bound_prec_den += bound_prec_
				bound_rec_den += bound_rec_
				
				tot_segments += 1
			
			tot_images += 1

print(f"Videos {tot_videos}")
print(f"Images {tot_images}")
print(f"Segments {tot_segments}")

print("Mask")
mask_prec = mask_inter/mask_prec_den
mask_rec = mask_inter/mask_rec_den
print(f"Precision: {mask_prec}")
print(f"Recall: {mask_rec}")
print(f"F-Score: {(2*mask_prec*mask_rec)/(mask_prec+mask_rec)}")

print("Boundary")
bound_prec = bound_inter/bound_prec_den
bound_rec = bound_inter/bound_rec_den
print(f"Precision: {bound_prec}")
print(f"Recall: {bound_rec}")
print(f"F-Score: {(2*bound_prec*bound_rec)/(bound_prec+bound_rec)}")