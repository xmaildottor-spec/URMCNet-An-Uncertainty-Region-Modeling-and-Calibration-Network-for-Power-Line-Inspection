# coding: utf-8
import os
import logging
from typing import List, Tuple, Optional, Callable

import imageio.v2 as imageio
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

# Retained to prevent potential OpenMP conflict issues
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class PLDataset(Dataset):
    """
    Dataset loader for Visual Light (VL), Infrared (IR), and Ground Truth (GT) masks.
    """

    def __init__(self, 
                 vl_list: List[str], 
                 ir_list: List[str], 
                 gt_list: List[str], 
                 transform: Optional[Callable] = None):
        super(PLDataset, self).__init__()

        self.vl_list = vl_list
        self.ir_list = ir_list
        self.gt_list = gt_list
        self.transform = transform
        self.n_data = len(self.vl_list)

        assert len(self.vl_list) == len(self.ir_list) == len(self.gt_list), \
            "All input lists must have the exact same length."

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor]:
        paths = {
            'vl': self.vl_list[index],
            'ir': self.ir_list[index],
            'gt': self.gt_list[index]
        }

        # Exception handling for missing or corrupted files
        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {name} file: {path}")

        try:
            vl = imageio.imread(paths['vl'])
            ir = imageio.imread(paths['ir'])
            gt = imageio.imread(paths['gt'])
        except Exception as e:
            raise RuntimeError(f"Failed to load image arrays at index {index}: {e}")

        # Apply external decoupled transformations
        if self.transform is not None:
            # Note: The transform pipeline should handle 'ir' as an additional target
            augmented = self.transform(image=vl, ir=ir, mask=gt)
            vl = augmented['image']
            ir = augmented['ir']
            gt = augmented['mask']
        else:
            vl = torch.from_numpy(vl.transpose(2, 0, 1)).float() / 255.0
            ir = torch.from_numpy(ir.transpose(2, 0, 1)).float() / 255.0
            gt = torch.from_numpy(gt).float().unsqueeze(0)

        # Convert mask to 2-channel format (Background, Foreground)
        gt = torch.cat([gt < 0.5, gt >= 0.5], dim=0).to(torch.float32)

        return vl, ir, gt

    def __len__(self) -> int:
        return self.n_data


class PLFPFNDataset(Dataset):
    """
    Dataset loader for Visual Light (VL), Infrared (IR), and multiple masks (GT, FNGT, FPGT).
    """

    def __init__(self, 
                 vl_list: List[str], 
                 ir_list: List[str], 
                 gt_list: List[str], 
                 fngt_list: List[str], 
                 fpgt_list: List[str], 
                 transform: Optional[Callable] = None):
        super(PLFPFNDataset, self).__init__()

        self.vl_list = vl_list
        self.ir_list = ir_list
        self.gt_list = gt_list
        self.fngt_list = fngt_list
        self.fpgt_list = fpgt_list
        self.transform = transform
        self.n_data = len(self.vl_list)

        assert len(self.vl_list) == len(self.ir_list) == len(self.gt_list) == \
               len(self.fngt_list) == len(self.fpgt_list), \
            "All input lists must have the exact same length."

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        paths = {
            'vl': self.vl_list[index],
            'ir': self.ir_list[index],
            'gt': self.gt_list[index],
            'fngt': self.fngt_list[index],
            'fpgt': self.fpgt_list[index]
        }

        for name, path in paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {name} file: {path}")

        try:
            vl = imageio.imread(paths['vl'])
            ir = imageio.imread(paths['ir'])
            gt = imageio.imread(paths['gt'])
            fngt = imageio.imread(paths['fngt'])
            fpgt = imageio.imread(paths['fpgt'])
        except Exception as e:
            raise RuntimeError(f"Failed to load image arrays at index {index}: {e}")

        if self.transform is not None:
            augmented = self.transform(image=vl, ir=ir, masks=[gt, fngt, fpgt])
            vl = augmented['image']
            ir = augmented['ir']
            gt, fngt, fpgt = augmented['masks']
        else:
            vl = torch.from_numpy(vl.transpose(2, 0, 1)).float() / 255.0
            ir = torch.from_numpy(ir.transpose(2, 0, 1)).float() / 255.0
            gt = torch.from_numpy(gt).float().unsqueeze(0)
            fngt = torch.from_numpy(fngt).float().unsqueeze(0)
            fpgt = torch.from_numpy(fpgt).float().unsqueeze(0)

        # Convert all masks to 2-channel format
        gt = torch.cat([gt < 0.5, gt >= 0.5], dim=0).to(torch.float32)
        fngt = torch.cat([fngt < 0.5, fngt >= 0.5], dim=0).to(torch.float32)
        fpgt = torch.cat([fpgt < 0.5, fpgt >= 0.5], dim=0).to(torch.float32)

        return vl, ir, gt, fngt, fpgt

        '''
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        import cv2 as cv
        import numpy as np
        import imgaug.augmenters as iaa

        class CustomVLAugmentation(A.ImageOnlyTransform):
            """
            Applies the custom HSV and imgaug weather corruptions specifically to the VL image.
            """
            def __init__(self, always_apply=False, p=0.5):
                super(CustomVLAugmentation, self).__init__(always_apply, p)
                self.fog_aug = iaa.imgcorruptlike.Fog(severity=1)
                self.snow_aug = iaa.imgcorruptlike.Snow(severity=1)

            def apply(self, img, **params):
                # img is passed in as RGB
                image_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                aug_idx = np.random.randint(4)

                if aug_idx == 0:
                    h, v, s = 1.0, 0.6, 0.7   
                    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV) 
                    hsv[:, :, 2] = (hsv[:, :, 2] * v).astype('uint8')
                    hsv[:, :, 1] = (hsv[:, :, 1] * s).astype('uint8') 
                    im_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR) 
                    im_bgr[:, :, 1] = cv.add(im_bgr[:, :, 1], 15)
                    im_bgr[:, :, 2] = cv.add(im_bgr[:, :, 2], 15)
                    img = im_bgr[:, :, ::-1] 

                elif aug_idx == 1:
                    h, v, s = 1.0, 0.3, 0.3
                    hsv = cv.cvtColor(image_bgr, cv.COLOR_BGR2HSV)
                    hsv[:, :, 2] = (hsv[:, :, 2] * v).astype('uint8')
                    hsv[:, :, 1] = (hsv[:, :, 1] * s).astype('uint8')
                    im_bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
                    im_bgr[:, :, 0] = cv.add(im_bgr[:, :, 0], 10)
                    img = im_bgr[:, :, ::-1]

                elif aug_idx == 2: 
                    img = self.fog_aug(images=[img])[0]

                elif aug_idx == 3: 
                    img = self.snow_aug(images=[img])[0]

                return img

        # --- How to use the decoupled pipeline in your training script ---

        train_transform = A.Compose(
            [
                # Apply your custom VL augmentations with 50% probability
                CustomVLAugmentation(p=0.5),
                
                # Standard synchronized augmentations (applied to VL, IR, and all masks identically)
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.Rotate(limit=15, p=0.5, interpolation=cv.INTER_LINEAR, border_mode=cv.BORDER_CONSTANT),
                
                # Normalization and Tensor Conversion
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ],
            # Tell Albumentations to treat the 'ir' input exactly like an image
            additional_targets={'ir': 'image'}
        )

        # IR Needs custom normalization ([0.5], [0.5]) as per your original code. 
        # Albumentations normalizes all 'image' targets identically. 
        # To handle IR's specific [0.5] norm, apply it manually after the pipeline or use a custom Transform.
        '''


    def __len__(self) -> int:
        return self.n_data