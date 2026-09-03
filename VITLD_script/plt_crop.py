import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

def calculate_plt(pred, labels, save_dir, iou_threshold=0.5):
    """
    Generates and saves the final prediction maps as individual images.
    The original batch concatenation logic has been replaced to save each image separately.
    
    Args:
        pred (torch.Tensor): Raw network predictions.
        labels (torch.Tensor): Ground truth labels.
        save_dir (str): Target directory for saving output images.
        iou_threshold (float): Threshold for binarizing predictions (default: 0.5).
    """
    os.makedirs(save_dir, exist_ok=True)

    out = nn.Sigmoid()
    pred = out(pred)

    pred = pred.data.cpu().numpy() 
    labels = labels.data.cpu().numpy()

    batch_size = pred.shape[0]

    start_idx = len([f for f in os.listdir(save_dir) if f.endswith('.png')])

    for i in range(batch_size):
        p = pred[i, 0, :, :]
        l = labels[i, 0, :, :]

        result_image = np.zeros((p.shape[0], p.shape[1], 3), dtype=np.uint8)
        
        result_image[(p >= iou_threshold) & (l == 1)] = [255, 255, 255]  # True Positive (TP): White
        result_image[(p >= iou_threshold) & (l == 0)] = [255, 255, 255]  # False Positive (FP): White
        # result_image[(p < iou_threshold) & (l == 1)]  = [0, 0, 255]    # False Negative (FN): Blue (Disabled)

        result_image = Image.fromarray(result_image)
        
        current_idx = start_idx + i
        result_filename = os.path.join(save_dir, f'{current_idx:04d}.png')
        result_image.save(result_filename)

    print(f"Saved {batch_size} individual prediction results from the current batch to: {save_dir}")


def calculate_plt_blue_red(pred, labels, save_dir, iou_threshold=0.5):
    """
    Generates and saves diagnostic prediction maps, highlighting false negatives and false positives.
    Each image is saved individually rather than as a concatenated grid.
    
    Color Coding:
    - True Positives (TP): White
    - False Positives (FP): Red
    - False Negatives (FN): Blue
    """
    os.makedirs(save_dir, exist_ok=True)

    out = nn.Sigmoid()
    pred = out(pred)

    pred = pred.data.cpu().numpy() 
    labels = labels.data.cpu().numpy()

    batch_size = pred.shape[0]

    start_idx = len([f for f in os.listdir(save_dir) if f.endswith('.png')])

    for i in range(batch_size):
        p = pred[i, 0, :, :]
        l = labels[i, 0, :, :]

        result_image = np.zeros((p.shape[0], p.shape[1], 3), dtype=np.uint8)
        
        result_image[(p >= iou_threshold) & (l == 1)] = [255, 255, 255]  # True Positive (TP): White
        result_image[(p >= iou_threshold) & (l == 0)] = [255, 0, 0]      # False Positive (FP): Red
        result_image[(p < iou_threshold)  & (l == 1)] = [0, 0, 255]      # False Negative (FN): Blue

        result_image = Image.fromarray(result_image)
        
        current_idx = start_idx + i
        result_filename = os.path.join(save_dir, f'{current_idx:04d}.png')
        result_image.save(result_filename)

    print(f"Saved {batch_size} individual diagnostic results from the current batch to: {save_dir}")