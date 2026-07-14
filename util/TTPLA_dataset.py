# coding: utf-8
import os
import logging
from typing import List, Tuple, Optional, Callable, Dict, Any

import imageio.v2 as imageio
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

class TTPLADataset(Dataset):
    """
    Standard dataset loader for visual light images and ground truth masks.
    """

    def __init__(self, vl_list: List[str], gt_list: List[str], transform: Optional[Callable] = None):
        super(TTPLADataset, self).__init__()
        self.vl_list = vl_list
        self.gt_list = gt_list
        self.transform = transform
        self.n_data = len(self.vl_list)

        assert len(self.vl_list) == len(self.gt_list), "Image and mask lists must have the same length."

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        vl_path = self.vl_list[index]
        gt_path = self.gt_list[index]

        # Exception handling for missing or corrupted files
        if not os.path.exists(vl_path) or not os.path.exists(gt_path):
            logging.error(f"File not found. VL: {vl_path}, GT: {gt_path}")
            raise FileNotFoundError(f"Missing data at index {index}")

        try:
            vl = imageio.imread(vl_path)
            gt = imageio.imread(gt_path)
        except Exception as e:
            logging.error(f"Error reading data at index {index}: {e}")
            raise

        # Apply decoupled transformations
        if self.transform is not None:
            augmented = self.transform(image=vl, mask=gt)
            vl = augmented['image']
            gt = augmented['mask']
        else:
            # Fallback to simple tensor conversion if no transforms are provided
            vl = torch.from_numpy(vl.transpose(2, 0, 1)).float() / 255.0
            gt = torch.from_numpy(gt).float().unsqueeze(0)

        # Ensure ground truth is binarized (0.0 or 1.0)
        gt = (gt >= 0.5).to(torch.float32)

        return vl, gt

    def __len__(self) -> int:
        return self.n_data


class TTPLAFPFNDataset(Dataset):
    """
    Dataset loader for multimodal inputs: images and multiple target masks.
    """

    def __init__(self, 
                 vl_list: List[str], 
                 gt_list: List[str], 
                 fngt_list: List[str], 
                 fpgt_list: List[str], 
                 transform: Optional[Callable] = None):
        super(TTPLAFPFNDataset, self).__init__()

        self.vl_list = vl_list
        self.gt_list = gt_list
        self.fngt_list = fngt_list
        self.fpgt_list = fpgt_list
        self.transform = transform
        self.n_data = len(self.vl_list)

        assert len(self.vl_list) == len(self.gt_list) == len(self.fngt_list) == len(self.fpgt_list), \
            "All input lists must have the exact same length."

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        paths = {
            'vl': self.vl_list[index],
            'gt': self.gt_list[index],
            'fngt': self.fngt_list[index],
            'fpgt': self.fpgt_list[index]
        }

        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {name} file: {path}")

        try:
            vl = imageio.imread(paths['vl'])
            gt = imageio.imread(paths['gt'])
            fngt = imageio.imread(paths['fngt'])
            fpgt = imageio.imread(paths['fpgt'])
        except Exception as e:
            raise RuntimeError(f"Failed to load image arrays at index {index}: {e}")

        # Albumentations handles multiple masks effortlessly using a list
        if self.transform is not None:
            augmented = self.transform(image=vl, masks=[gt, fngt, fpgt])
            vl = augmented['image']
            gt, fngt, fpgt = augmented['masks']
        else:
            vl = torch.from_numpy(vl.transpose(2, 0, 1)).float() / 255.0
            gt = torch.from_numpy(gt).float()
            fngt = torch.from_numpy(fngt).float()
            fpgt = torch.from_numpy(fpgt).float()

        # Add channel dimensions to masks and binarize
        if gt.ndim == 2: gt = gt.unsqueeze(0)
        if fngt.ndim == 2: fngt = fngt.unsqueeze(0)
        if fpgt.ndim == 2: fpgt = fpgt.unsqueeze(0)

        gt = (gt >= 0.5).to(torch.float32)
        fngt = (fngt >= 0.5).to(torch.float32)
        fpgt = (fpgt >= 0.5).to(torch.float32)

        return vl, gt, fngt, fpgt

        '''
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Define the training augmentation pipeline
        train_transform = A.Compose([
            A.RandomCrop(width=512, height=512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.Rotate(limit=15, p=0.5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Define the testing augmentation pipeline (no random flips/crops)
        test_transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        # Instantiate the dataset
        train_dataset = TTPLAFPFNDataset(
            vl_list=train_vl_paths,
            gt_list=train_gt_paths,
            fngt_list=train_fngt_paths,
            fpgt_list=train_fpgt_paths,
            transform=train_transform)
        '''

    def __len__(self) -> int:
        return self.n_data