# for coco, copy the initial pseudo-labels to semicoco dir
# anno_path="/data/semicoco/unlabel_prepared_annos/Industry/annotations/full/"
# rm -rf $anno_path
# cp -r workdir_coco/r50_caffe_mslonger_tricks_0.1data/epoch_55.pth-unlabeled.bbox.json_thres0.1_annos/ $anno_path

# for voc, copy the initial pseudo-labels to semivoc dir
rm -rf ../data/semivoc/unlabel_prepared_annos/Industry/annotations/full/
cp -r workdir_voc/voc_coco/r50/epoch_60.pth-unlabeled.bbox.json_thres0.1_annos_merged/ ../data/semivoc/unlabel_prepared_annos/Industry/annotations/full/
echo "remove & copy annotations done!"

CONFIG=configs/fcos_semi/voc/RLA_anchor-free.py
WORKDIR=workdir_voc/voc_coco/dy/07train+12trainval/label-merged/thresh0.1/r1

GPU=1

CUDA_VISIBLE_DEVICES=0 PORT=29502 ./tools/dist_train.sh $CONFIG $GPU --work-dir $WORKDIR
