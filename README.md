# Paper Code Release

## Environment Configuration

### Semi-Supervised Framework

The environment for the main semi-supervised framework is configured according to the instructions in the [DSL repository](https://github.com/chenbinghui1/dsl). Please follow the setup instructions provided there to install the necessary dependencies and configure your environment.

### OOD Detector

For the OOD detector, both the environment and the dataset are configured as detailed in the [OpenDet2 repository](https://github.com/csuhan/opendet2). Refer to that repository for guidance on environment setup, dependency installation, and dataset preparation.

## Procedure

### 1. Download VOC data
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

### 2. Convert data format as [DSL repository](https://github.com/chenbinghui1/dsl) did

### 3. Train as steps4-steps7 which are used in Partially Labeled data protocol

## Testing

To test the model, run the following command:

```bash
python ./test.py
