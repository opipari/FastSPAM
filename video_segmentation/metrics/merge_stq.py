import os
import torch

epsilon=1e-15
root = "/home/ANT.AMAZON.COM/topipari/Downloads/evaluate_stq/"
print(os.listdir(root))

for moddir in os.listdir(root):
	data010 = torch.load(os.path.join(root,moddir,"res0100.json"))
	data100 = torch.load(os.path.join(root,moddir,"res74.json"))
	# print(moddir)
	out_data = {}
	out_data["aq_per_seq"] = torch.cat([data010["aq_per_seq"],data100["aq_per_seq"]])
	out_data["num_tubes_per_seq"] = torch.cat([data010["num_tubes_per_seq"],data100["num_tubes_per_seq"]])
	out_data["iou_per_seq"] = torch.cat([data010["iou_per_seq"],data100["iou_per_seq"]])
	out_data["stq_per_seq"] = torch.cat([data010["stq_per_seq"],data100["stq_per_seq"]])
	out_data["class_intersections"] = data010["class_intersections"]+data100["class_intersections"]
	out_data["class_unions"] = data010["class_unions"]+data100["class_unions"]
	
	aq_mean = out_data["aq_per_seq"].sum() / torch.clamp(out_data["num_tubes_per_seq"].sum(), min=epsilon)
	num_classes_nonzero = len(out_data["class_unions"].nonzero())
	ious = out_data["class_intersections"] / torch.clamp(out_data["class_unions"], min=epsilon)
	iou_mean =  ious.sum() / num_classes_nonzero

	stq = torch.sqrt(aq_mean * iou_mean)

	out_data["stq"] = stq.item()
	out_data["aq"] = aq_mean.item()
	out_data["iou"] = iou_mean.item()

	torch.save(out_data, os.path.join(root,moddir,"res.json"))
	print(os.path.join(root,moddir,"res.json"))