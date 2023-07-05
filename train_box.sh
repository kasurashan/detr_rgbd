# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env main.py \
# --masks \
# --epochs 50  \
# --lr_drop 15 \
# --coco_path ../../datasets/nyuv2  \
# --resume ./detr-r50-panoptic-00ce5173.pth \
# --output_dir ./output/RGBD_nyu_2way_DrSr_50ep_15drop_new \
# --depth_data nyu \
# --head_feature 'DrSr'
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env main.py \
# --masks \
# --epochs 50  \
# --lr_drop 15 \
# --coco_path ../../datasets/nyuv2  \
# --resume ./detr-r50-panoptic-00ce5173.pth \
# --output_dir ./output/RGBD_nyu_2way_DrSf_50ep_15drop_new \
# --depth_data nyu \
# --head_feature 'DrSf'
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env main.py \
# --masks \
# --epochs 50  \
# --lr_drop 15 \
# --coco_path ../../datasets/nyuv2  \
# --resume ./detr-r50-panoptic-00ce5173.pth \
# --output_dir ./output/RGBD_nyu_2way_DfSr_50ep_15drop_new \
# --depth_data nyu \
# --head_feature 'DfSr'

# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env main.py \
# --masks \
# --epochs 200  \
# --lr_drop 130 \
# --coco_path ../../datasets/boxdata/BOX_DATA/  \
# --resume ./detr-r50-panoptic-00ce5173.pth \
# --output_dir ./output/RGBD_nyu_2way_DrSf_200ep_130drop_new \
# --depth_data box \
# --head_feature 'DrSf'
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env main.py \
# --masks \
# --epochs 200  \
# --lr_drop 130 \
# --coco_path ../../datasets/boxdata/BOX_DATA/  \
# --resume ./detr-r50-panoptic-00ce5173.pth \
# --output_dir ./output/RGBD_nyu_2way_DfSr_200ep_130drop_new \
# --depth_data box \
# --head_feature 'DfSr'
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env main.py \
# --masks \
# --epochs 200  \
# --lr_drop 130 \
# --coco_path ../../datasets/boxdata/BOX_DATA/  \
# --resume ./detr-r50-panoptic-00ce5173.pth \
# --output_dir ./output/RGBD_box_2way_DrSf_200ep_130drop_new_poslearn \
# --depth_data box \
# --position_embedding 'sine'
# --head_feature 'DrSf'
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env main.py \
# --masks \
# --epochs 200  \
# --lr_drop 130 \
# --coco_path ../../datasets/boxdata/BOX_DATA/  \
# --resume ./detr-r50-panoptic-00ce5173.pth \
# --output_dir ./output/RGBD_box_concat_200ep_130drop \
# --depth_data box \
# --head_feature 'concat'
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env ../detr/main.py \
# --masks \
# --epochs 50 \
# --lr_drop 130 \
# --coco_path ../../datasets/nyuv2/  \
# --output_dir ../detr/output/RGB_nyu_50ep_drop1040 \
# --resume ../detr/detr-r50-panoptic-00ce5173.pth


CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 \
--use_env main.py \
--masks \
--epochs 200  \
--lr_drop 130 \
--coco_path ../../datasets/boxdata/BOX_DATA/  \
--resume /root/datasets/output_detr_rgbd/RGBD_nyu_2way_DrSf_50ep_15drop_new/checkpoint_best_segm.pth \
--output_dir ../../datasets/output_detr_rgbd \
--depth_data box \
--resume_reset True \
--head_feature 'DrSf'



# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env main.py \
# --masks \
# --epochs 200  \
# --lr_drop 130 \
# --coco_path ../../datasets/boxdata/BOX_DATA/  \
# --resume ./detr-r50-panoptic-00ce5173.pth \
# --output_dir ./output/RGBD_box_2way_DfSf_200ep_newrgb_newd \
# --depth_data box \
# --head_feature 'DfSf'
# # --lr_backbone 5e-6 \
# # --lr 5e-5



# ########detection용 ()
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env main.py \
# --epochs 200  \
# --lr_drop 130 \
# --coco_path ../../datasets/boxdata/BOX_DATA/  \
# --resume ./detr-r50-e632da11.pth \
# --output_dir ./output/RGBD_box_2way_DfSf_200ep_detection_enrichRGB \
# --depth_data box \
# --head_feature 'DfSf'
# #######detection용 ()
# CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
# --nproc_per_node=2 \
# --use_env main.py \
# --epochs 50  \
# --lr_drop 15 \
# --coco_path ../../datasets/nyuv2/  \
# --resume ./detr-r50-e632da11.pth \
# --output_dir ./output/RGBD_nyu_2way_DfSf_50ep_detection_enrichRGB \
# --depth_data nyu \
# --head_feature 'DfSf'
########detection용 ()