# Paper Code Release

## Environment Configuration

### Semi-Supervised Framework

```bash
pytorch>=1.8.0
cuda 10.2
python>=3.8
mmcv-full 1.3.10
```

### OOD Detector

```
python=3.8
pytorch=1.8.1
detectron2==0.5
```

## Procedure

### I. Get original pseudo-labels

#### 1. Download VOC data and coco dataset
Download VOC dataset to dir xx and unzip it, we will get (`VOCdevkit/`)
```bash
# resulting format
# ori_data/
#   - VOCdevkit
#     - VOC2007
#       - Annotations
#       - JPEGImages
#       - ...
#     - VOC2012
#       - Annotations
#       - JPEGImages
#       - ...
```
Download coco dataset and unzip it

```bash
# resulting format
# ori_data/coco
#   - train2017
#     - xxx.jpg
#   - val2017
#     - xxx.jpg
#   - unlabled2017
#     - xxx.jpg
#   - annotations
#     - xxx.json
#     - ...
```

#### 2. Prepare data

- Copy the images and annotations from **VOC07 `train`** and **VOC12 `trainval`** into the corresponding directories of the **VOC2007** folder as the labeled training set.  
- Copy the images from **COCO `Unlabel`** into the corresponding directory of the **VOC2012** folder as the unlabeled training set.


#### 3. Train supervised baseline model as shown in DSL
```bash
cd DSL
./demo/model_train/baseline_voc.sh
```
#### 4. Generate initial pseudo-labels for unlabeled images(1/2)
Generate the initial pseudo-labels for unlabeled images via (`tools/inference_unlabeled_coco_data.sh`): please change the corresponding list file path of unlabeled data in the config file, and the model path in tools/inference_unlabeled_coco_data.sh.
```bash
./tools/inference_unlabeled_coco_data.sh
```

Then you will obtain (`workdir_coco/xx/epoch_xxx.pth-unlabeled.bbox.json`) which contains the pseudo-labels.

#### 5. Generate initial pseudo-labels for unlabeled images(2/2)
Use (`tools/generate_unlabel_annos_coco.py`) to convert the produced (`epoch_xxx.pth-unlabeled.bbox.json`) above to DSL-style annotations
```bash
python3 tools/generate_unlabel_annos_coco.py --input_path workdir_coco/xx/epoch_xxx.pth-unlabeled.bbox.json --input_list data_list/coco_semi/semi_supervised/instances_train2017.${seed}@${percent}-unlabeled.json --cat_info ${project_root_dir}/data/semicoco/mmdet_category_info.json --thres 0.1      
```

You will obtain (`workdir_coco/xx/epoch_xxx.pth-unlabeled.bbox.json_thres0.1_annos/`) dir which contains the original pseudo-labels.

### II. Get OOD pseudo-labels

#### 1. Prepare data as shown in OpenDet

#### 2. Training OOD detector

The training process is the same as detectron2.
```
python opendet/tools/train_net.py --num-gpus 2 --config-file configs/faster_rcnn_R_50_FPN_3x_opendet.yaml
```

#### 3. Produce OOD pseudo-labels

- Replace the original file at `/home/anaconda3/envs/opendet/lib/python3.8/site-packages/detectron2/evaluation/evaluator.py` with the `evaluator.py` from this repository.

- Then, run the following command:

  ```bash
  python tools/train_net.py --num-gpus 2 --config-file configs/faster_rcnn_R_50_FPN_3x_opendet.yaml \
  MODEL.WEIGHTS output/faster_rcnn_R_50_FPN_3x_opendet/model_final.pth
   ```
The test dataset should be the one you want to perform predictions on, it should be the images from **COCO `Unlabel`**. After execution, pseudo-labels for **COCO `Unlabel`** part will be generated.

### III. Hybrid lables

#### 1.Mix the two kinds of labels above

```bash
  python tools/merge_label.py
```

Then replace original labels with hybrid labels

### IV. Training

```bash
./model_train/semi-voc.sh
```


