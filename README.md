# Paper Code Release

## Environment Configuration

### Semi-Supervised Framework

The environment for the main semi-supervised framework is configured according to the instructions in the [DSL](https://github.com/chenbinghui1/dsl). Please follow the setup instructions provided there to install the necessary dependencies and configure your environment.

### OOD Detector

We use opendet as OOD detector, both the environment and the dataset are configured as detailed in the [OpenDet](https://github.com/csuhan/opendet2). Refer to that repository for guidance on environment setup, dependency installation, and dataset preparation.

## Procedure

### 1. Download VOC data and coco dataset
Download VOC dataset to dir xx and unzip it, we will get (`VOCdevkit/`)
```bash
cd ${project_root_dir}/ori_data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_06-Nov-2007.tar
tar -xf VOCtest_06-Nov-2007.tar
tar -xf VOCtrainval_11-May-2012.tar

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


```bash
mkdir ori_data/coco
cd ori_data/coco

wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/unlabeled2017.zip

unzip annotations_trainval2017.zip -d .
unzip -q train2017.zip -d .
unzip -q val2017.zip -d .
unzip -q unlabeled2017.zip -d .

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

### 2.Prepare data

Copy the images and annotations from **VOC07 `train`** and **VOC12 `trainval`** into the corresponding directories of the **VOC2007** folder as the labeled training set.  
Copy the images from **COCO `Unlabel`** into the corresponding directory of the **VOC2012** folder as the unlabeled training set.


### 2. Convert VOC and COCO data format respectively as [DSL](https://github.com/chenbinghui1/dsl) did

### 3. Train as steps4-steps7 which are used in Partially Labeled data protocol in [DSL](https://github.com/chenbinghui1/dsl)

## Testing

To test the model, run the following command:

```bash
python ./test.py
