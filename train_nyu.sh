CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--nproc_per_node=2 \
--use_env main.py \
--masks \
--epochs 200  \
--coco_path ../../datasets/nyuv2/  \
--resume ./detr-r50-panoptic-00ce5173.pth \
--output_dir ./output/test \
--depth_data nyu