# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os
from torchvision import datasets, transforms
import cv2
import json
from pathlib import Path
import torch.utils.data as data

import os
import time           # 解决 name 'time' is not defined
import cv2            # 解决 name 'cv2' is not defined
import numpy as np
from PIL import Image # 解决 name 'Image' is not defined
import random

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data import create_transform

def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'IMNET':
        print("reading from datapath", args.data_path)
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == "image_folder":
        root = args.data_path if is_train else args.eval_data_path
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = args.nb_classes
        assert len(dataset.class_to_idx) == nb_classes
    elif args.data_set == "json":
        root = args.data_path if is_train else args.eval_data_path
        imset = os.path.join(args.imset, 'train.json' if is_train else 'test.json')
        nb_classes = args.nb_classes
        dataset = JsonDataset(imset, root, transform=transform)
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % nb_classes)

    return dataset, nb_classes

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            scale=(0.8, 1.0),
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # warping (no cropping) when evaluated at 384 or larger
        if args.input_size >= 384:  
            t.append(
            transforms.Resize((args.input_size, args.input_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC), 
        )
            print(f"Warping {args.input_size} size input images...")
        else:
            if args.crop_pct is None:
                args.crop_pct = 224 / 256
            size = int(args.input_size / args.crop_pct)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),  
            )
            t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

class JsonDataset(data.Dataset):
    def __init__(self, imset, root, transform=None):
        self.transform=transform

        root = Path(root)
        self.to_pil = transforms.ToPILImage()

        with open(imset, 'r', encoding='utf8') as f:
            data = json.load(f)
            self.data_list=[(str(root/k), v) for k,v in data.items()]

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        
        max_retries = 10
        img = None
        
        for i in range(max_retries):
            try:
                # 增强型读取：支持中文、空格和特殊路径
                raw_data = np.fromfile(img_path, dtype=np.uint8)
                raw_img = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
                
                if raw_img is not None:
                    # 转换为 PIL 对象以适配后续 transform
                    img = Image.fromarray(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))
                    break
            except Exception as e:
                # 记录读取失败，方便后期手动清理
                print(f"[Rank0] 读取失败 (第 {i+1}/10 次): {img_path}, 错误: {e}")
                time.sleep(0.1 * (2 ** i) if i < 5 else 1.0)

        # --- 核心改进：如果不成功，随机换一张图再读一次 ---
        if img is None:
            print(f"!!! 警告：尝试 10 次仍失败。正在随机替换图片读取: {img_path}")
            # 在整个数据集里随机挑一个新索引
            new_idx = random.randint(0, len(self.data_list) - 1)
            # 递归调用自己，直到读到一张好图为止
            return self.__getitem__(new_idx)

        # 正常执行数据增强
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data_list)

    def get_classes_for_all_imgs(self):
        return [x[1] for x in self.data_list]
