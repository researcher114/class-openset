import os
import json
import math

folder1_path = '/home/opendet/opendet2/inference-result/voc_coco/07train+12trainval/'
folder2_path = '/home/DSL/workdir_voc/voc_coco/epoch_60.pth-unlabeled.bbox.json_thres0.1_annos/'

folder1_files = os.listdir(folder1_path)
folder2_files = os.listdir(folder2_path)

GAMMA = 1.5

for filename in folder1_files:
    if filename in folder2_files:
        with open(os.path.join(folder1_path, filename), 'r') as f1, \
             open(os.path.join(folder2_path, filename), 'r') as f2:
            data1 = json.load(f1)
            data2 = json.load(f2)

        unknown_tags = []
        unknown_rects = []

        for tag, score, rect in zip(data1["tags"], data1["scores"], data1["rects"]):
            if tag == "unknown" and score > 0.6:
               
                Ti = max(0, min(1, math.exp(GAMMA * (score - 1))))
                if Ti > 0.5:
                    unknown_tags.append(score)
                    unknown_rects.append(rect)

        data2["tags"].extend(["unknown"] * len(unknown_tags))
        data2["scores"].extend(unknown_tags)
        data2["rects"].extend(unknown_rects)
        data2["masks"].extend([[]] * len(unknown_tags))
        data2["targetNum"] += len(unknown_tags)

        folder_path = '/home/DSL/workdir_voc/voc_coco/RetinaNet/epoch_60.pth-unlabeled.bbox.json_thres0.1_annos_merged/'

        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        merged_filename = os.path.join(folder_path, filename)

        with open(merged_filename, 'w', encoding='utf-8') as merged_file:
            json.dump(data2, merged_file, ensure_ascii=False, indent=4)

        print(f"save {merged_filename}")
