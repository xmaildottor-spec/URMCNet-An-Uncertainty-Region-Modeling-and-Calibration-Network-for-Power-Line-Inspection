import os
import numpy as np
from PIL import Image
import torch.nn as nn

def calculate_plt(pred, labels, save_dir, img_index, iou_threshold=0.5):
    """
    Generates and saves the final prediction maps by stitching four 256x256 patches into a single 512x512 image.
    
    Args:
        pred (torch.Tensor): Raw network predictions for the batch.
        labels (torch.Tensor): Ground truth labels for the batch.
        save_dir (str): Target directory for saving output images.
        img_index (int): Index for sequentially naming the output file.
        iou_threshold (float): Threshold for binarizing predictions (default: 0.5).
    """
    # Ensure the target directory exists
    os.makedirs(save_dir, exist_ok=True)

    out = nn.Sigmoid()
    pred = out(pred)

    pred = pred.data.cpu().numpy() 
    labels = labels.data.cpu().numpy()

    com_pred = np.zeros([512, 512])
    com_lable = np.zeros([512, 512])
    
    # Predictions and labels are 1-channel; thus, the channel index is set to 0
    com_pred[:256, :256] = pred[0, 0, :, :]
    com_pred[:256, 256:] = pred[1, 0, :, :]
    com_pred[256:, :256] = pred[2, 0, :, :]
    com_pred[256:, 256:] = pred[3, 0, :, :]
    
    com_lable[:256, :256] = labels[0, 0, :, :]
    com_lable[:256, 256:] = labels[1, 0, :, :]
    com_lable[256:, :256] = labels[2, 0, :, :]
    com_lable[256:, 256:] = labels[3, 0, :, :]   

    result_image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Apply color mapping based on prediction results
    result_image[(com_pred >= iou_threshold) & (com_lable == 1)] = [255, 255, 255]  # True Positive (TP): White
    result_image[(com_pred >= iou_threshold) & (com_lable == 0)] = [255, 255, 255]  # False Positive (FP): White
    # False Negative (FN): Remains background (black)

    result_image = Image.fromarray(result_image)
    
    # Use 04d formatting to generate filenames sequentially (e.g., 0000.png)
    result_filename = os.path.join(save_dir, f'{img_index:04d}.png')
    result_image.save(result_filename)

    print(f"Result successfully generated and saved to: {result_filename}")


def calculate_plt_blue_red(pred, labels, save_dir, img_index, iou_threshold=0.5):
    """
    Generates and saves the final diagnostic prediction maps with distinct colors for false negatives and false positives.
    Stitches four 256x256 patches into a single 512x512 image.
    
    Args:
        pred (torch.Tensor): Raw network predictions for the batch.
        labels (torch.Tensor): Ground truth labels for the batch.
        save_dir (str): Target directory for saving output images.
        img_index (int): Index for sequentially naming the output file.
        iou_threshold (float): Threshold for binarizing predictions (default: 0.5).
    """
    # Ensure the target directory exists
    os.makedirs(save_dir, exist_ok=True)

    out = nn.Sigmoid()
    pred = out(pred)

    pred = pred.data.cpu().numpy() 
    labels = labels.data.cpu().numpy()

    com_pred = np.zeros([512, 512])
    com_lable = np.zeros([512, 512])
    
    # Predictions and labels are 1-channel; thus, the channel index is set to 0
    com_pred[:256, :256] = pred[0, 0, :, :]
    com_pred[:256, 256:] = pred[1, 0, :, :]
    com_pred[256:, :256] = pred[2, 0, :, :]
    com_pred[256:, 256:] = pred[3, 0, :, :]
    
    com_lable[:256, :256] = labels[0, 0, :, :]
    com_lable[:256, 256:] = labels[1, 0, :, :]
    com_lable[256:, :256] = labels[2, 0, :, :]
    com_lable[256:, 256:] = labels[3, 0, :, :]   

    result_image = np.zeros((512, 512, 3), dtype=np.uint8)
    
    # Apply color mapping using the stitched com_pred and com_lable arrays
    result_image[(com_pred >= iou_threshold) & (com_lable == 1)] = [255, 255, 255]  # True Positive (TP): White
    result_image[(com_pred >= iou_threshold) & (com_lable == 0)] = [255, 0, 0]      # False Positive (FP): Red
    result_image[(com_pred < iou_threshold) & (com_lable == 1)] = [0, 0, 255]       # False Negative (FN): Blue

    result_image = Image.fromarray(result_image)
    
    # Use 04d formatting to generate filenames sequentially (e.g., 0000.png)
    result_filename = os.path.join(save_dir, f'{img_index:04d}.png')
    result_image.save(result_filename)

    print(f"Result successfully generated and saved to: {result_filename}")