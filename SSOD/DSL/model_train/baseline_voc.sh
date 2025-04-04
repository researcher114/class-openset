#!/usr/bin/env bash

#CONFIG=$1
#GPUS=$2
#PORT=${PORT:-29500}

#PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
#python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}
CONFIG=configs/r50_anchor-free.py
WORKDIR=workdir_voc/voc_coco/r50
# WORKDIR=workdir_voc/voc_coco/test
GPU=1

CUDA_VISIBLE_DEVICES=0 PORT=29505 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR
