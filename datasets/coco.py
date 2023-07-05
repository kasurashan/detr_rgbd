# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import datasets.transforms as T


import numpy as np   # depth npy load
from torch.utils.data import Dataset
import glob
import os
from PIL import Image
from matplotlib import pyplot as plt
import random
from time import time


class CocoDetection(torchvision.datasets.CocoDetection):
    # original
    # def __init__(self, img_folder, ann_file, transforms, return_masks):
    #     super(CocoDetection, self).__init__(img_folder, ann_file)
    #     self._transforms = transforms
    #     self.prepare = ConvertCocoPolysToMask(return_masks)
    
    # def __getitem__(self, idx):
    #     img, target = super(CocoDetection, self).__getitem__(idx)
    #     image_id = self.ids[idx]
    #     target = {'image_id': image_id, 'annotations': target}
    #     img, target = self.prepare(img, target)
    #     if self._transforms is not None:
    #         img, target = self._transforms(img, target)
    #     return img, target


    # add depth path  and depth_data type      
    def __init__(self, img_folder, ann_file, depth_folder, transforms, return_masks, depth_data):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        
        #self.file_pattern = "*.npy"
        self.depth_folder = depth_folder
        #self.depth_files = glob.glob(os.path.join(self.depth_folder, self.file_pattern))
        self.depth_data = depth_data
        
    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        #print(self.ids)
        #print(idx) # idx : not sorted
        #print(self.coco.loadImgs(idx)[0]['file_name'])
        
        
        
        
        image_id = self.ids[idx]               
        img_file_name = self.coco.loadImgs(image_id)[0]['file_name']
        
        if self.depth_data == 'box':
            depth_file = img_file_name.rstrip('.png') + '.npy'
            depth_data_path = str(self.depth_folder) + '/' + depth_file
            depth_data = np.load(depth_data_path)
            try:
                depth_data = depth_data.squeeze()   #depth data에 [1,1,h,w]크기인 것도 섞여있어서 전부 [h,w]로 맞춤
            except:
                pass
            depth_img = Image.fromarray(depth_data, mode='F')   # mode = F 로 해야 floating point로 받음
            
        if self.depth_data == 'nyu':
            #print(img_file_name)   # train/color/nyu_rgb_0816.png  
            depth_file = '00' + img_file_name.lstrip('train/color/nyu_rgb_').lstrip('val/color/nyu_rgb_')
            
            #print(depth_file, 1111111111111)
            depth_data_path = str(self.depth_folder) + '/' + depth_file
            depth_data = Image.open(depth_data_path)
            #print(np.array(depth_data).shape)   # (480 640)
            #print(np.array(depth_data))   # 원소가 정수형으로 1990, 1040, 2000 이런식인ㄷㅅ
            #print(np.array(depth_data))
            #depth_img = Image.fromarray(np.array(depth_data), mode='F')   # float으로
            depth_img = np.array(depth_data).astype(np.float32)
            depth_img = depth_img / np.max(depth_img) * 255
            depth_img = Image.fromarray(depth_img)
            # print(np.array(depth_img))
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        
        #depth_data_path = self.depth_files[idx]
        
        
        #print(depth_data_path)
        
        #print(1111111111111)
        #print(image_id, self.coco.loadImgs(image_id)[0]['file_name'], depth_file)
        
        
        
            
        #print(depth_data.shape)
        
        #depth_tensor = torch.from_numpy(depth_data).squeeze(dim=0)
        ##print(22222222222222222222222222222222222222222222222222)
        # plt.title(f'{depth_file}')   # depth와 img가 실제로 같은 것끼리 대응되는지 시각적으로 확인 (뒤집어졌는지 이런거)
        # plt.imshow(depth_data)
        # plt.savefig('/root/workspace/1.png')

        # plt.title(f'{img_file_name}')
        # plt.imshow(np.array(img))
        # plt.savefig('/root/workspace/2.png')
        
        ##print(333333333333333333333333333333333333333333333333333333333333)
        #depth_data = Image.fromarray(depth_data.astype(np.uint8)) 
        
        #print(self._transforms.transforms)
        #print(depth_data, 222222221)
        
        #print(depth_tensor.shape, 11111111111)
        
        # print(np.array(img).shape, depth_data.shape)    # shape :   [708,1280,3]  /   [708, 1280]
        
        #img = np.array(img)
        
        
        #depth_data = depth_data[:, :, np.newaxis]
        
        #depth_data = depth_data / (np.max(depth_data) - np.min(depth_data)) * 255
        
        #depth_data = np.concatenate((depth_data,depth_data,depth_data), axis=2)
        
        
        #print(depth_data)
        #print(np.uint8(depth_data))
        
        
        
        
        #print(np.array(depth_data), np.array(depth_data).shape)
        #depth_data = depth_data[:, :, np.newaxis]
        #img = np.concatenate((img, depth_data), axis=2)
        #print(img.shape) # [708,1280,4]
        #img = Image.fromarray(img.astype(np.uint8)) 
        
        
        
        #seed = np.random.randint(2147483647)
        #random.seed(seed)

        #img = torch.tensor(np.array(img))
        
        img = [img, depth_img]   #[rgb, depth]
        #print(np.array(img[0]), np.array(img[1]))
        if self._transforms is not None:
            img, target = self._transforms(img, target)    # transforms.py를 수정하여 [rgb depth] 리스트를 받을 수 있도록 수정

        #random.seed(seed)  #랜덤하게 크롭하거나 리사이즈하는 확률이 rgb와 같도록 (안 먹히는듯)
        # if self._transforms2 is not None:
            #del self._transforms2.transforms[-1]
            #print(self._transforms2.transforms[-1].transforms)
            #print(self._transforms2)
            # del self._transforms2.transforms[-1].transforms[-1]   #
            #print(self._transforms2.transforms[-1], 111111111)
            
            # depth_data, _ = self._transforms2(depth_data, target)  # T.normalize 삭제한 transform
            
        
        #print(depth_data.shape, img.shape)        
        #print(img.shape, 111111111111)
        #print(np.array(depth_tensor).shape, 11111111111)
        # plt.title(f'{depth_file}')   # depth와 img가 실제로 같은 것끼리 대응되는지 시각적으로 확인 (뒤집어졌는지 이런거)
        # plt.imshow(img[1].squeeze())
        # plt.savefig(f'/root/workspace/testimg/1{time()}.png')

        # plt.title(f'{img_file_name}')
        # plt.imshow(img[0].mean(dim=0))
        # plt.savefig(f'/root/workspace/testimg/2{time()}.png')

        #img = torch.cat([img, depth_data], dim=0)
        #print(depth_data, img)

        
        #img = torch.tensor(np.array(img[0]) + np.array(img[1]))   #elementwise sum
        
        img = torch.tensor(np.concatenate((np.array(img[0]), np.array(img[1])), axis=0))   # [4, h, w]  concat해서 채널 4됨
        

        return img, target






def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')



def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    
    
    
    # original path
    # PATHS = {
    #     "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    #     "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    # }
    
    #img_folder, ann_file = PATHS[image_set]
    #dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)
    
    
    # add depth path
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json', root / "train2017_depth"),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json', root / "val2017_depth"),
    }

    img_folder, ann_file, depth_folder = PATHS[image_set]
    dataset = CocoDetection(img_folder, ann_file, depth_folder, transforms=make_coco_transforms(image_set),return_masks=args.masks, depth_data=args.depth_data)   # should edit COCODETECTION
    
    return dataset






def build_original(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection_original(img_folder, ann_file, transforms=None, return_masks=args.masks)
    return dataset

class CocoDetection_original(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection_original, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection_original, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target