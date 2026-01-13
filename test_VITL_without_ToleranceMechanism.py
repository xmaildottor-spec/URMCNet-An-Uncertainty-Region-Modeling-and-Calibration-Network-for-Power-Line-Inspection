"""
Evaluation Script: Standard Strict Mode (No Tolerance)
======================================================
Usage:
    python eval_standard.py --data_root "./dataset" --weights "./model.pth"
"""

import os
import argparse
import time
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import List, Tuple, Union

# ==============================================================================
# Custom Imports
# ==============================================================================
try:
    from setproctitle import setproctitle
    setproctitle("Evaluation_Standard")
except ImportError:
    pass

try:
    from util.PL_dataset_jpg import PL_dataset 
    from Your_model import Your_model 
except ImportError as e:
    PL_dataset = object 
    Your_model = object

# ==============================================================================
# Logging Configuration
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_sorted_files(directory: str) -> List[str]:
    """Retrieves a sorted list of image files from a directory."""
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    try:
        files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in valid_extensions]
        return sorted(files)
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory}")
        return []

# ==============================================================================
# Metric Calculation Utilities (Strict Mode)
# ==============================================================================

def calculate_standard_confusion(pred: torch.Tensor, 
                                 target: torch.Tensor, 
                                 threshold: float = 0.3) -> Tuple[int, int, int, int]:
    """
    Calculates STRICT confusion matrix elements (No Tolerance).
    
    Args:
        pred (torch.Tensor): Prediction tensor.
        target (torch.Tensor): Original Ground Truth.
        threshold (float): Binarization threshold.

    Returns:
        Tuple[int, int, int, int]: TP, FP, FN, TN.
    """
    # Dimension Handling
    if pred.dim() == 3 and pred.shape[0] == 2: pred = pred[1, :, :]
    elif pred.dim() == 3: pred = pred.squeeze(0)
            
    if target.dim() == 3: target = target.squeeze(0)

    # Binarize
    pred_prob = torch.sigmoid(pred)
    pred_mask = (pred_prob > threshold)
    target_mask = (target > 0)

    # Standard Strict Logic
    # ---------------------
    # TP: Prediction is 1 AND Target is 1
    tp = (pred_mask & target_mask).sum().item()

    # FP: Prediction is 1 BUT Target is 0
    fp = (pred_mask & (~target_mask)).sum().item()

    # FN: Prediction is 0 BUT Target is 1
    fn = ((~pred_mask) & target_mask).sum().item()

    # TN: Prediction is 0 AND Target is 0
    tn = ((~pred_mask) & (~target_mask)).sum().item()

    return tp, fp, fn, tn


# ==============================================================================
# Evaluation Loops
# ==============================================================================

def evaluate_micro_standard(model, loader, device):
    """
    Performs Micro-Average evaluation.
    Accumulates TP/FP/FN over the whole dataset first, then computes metrics.
    """
    model.eval()
    
    total_tp, total_fp, total_fn, total_tn = 0, 0, 0, 0
    epsilon = 1e-6
    
    logger.info("Starting Micro-Average Evaluation...")

    with torch.no_grad():
        for _, (input_vl, input_ir, labels) in enumerate(loader):
            input_vl = input_vl.to(device)
            labels = labels.to(device)[:, 1, :, :].unsqueeze(1)

            outputs = model(input_vl)
            pre_final = outputs[0] if isinstance(outputs, tuple) else outputs
            
            batch_size = pre_final.shape[0]

            for i in range(batch_size):
                tp, fp, fn, tn = calculate_standard_confusion(
                    pre_final[i], labels[i], threshold=0.3
                )
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

    # Metrics
    iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + epsilon)

    return iou, acc, f1, recall, precision


def evaluate_macro_standard(model, loader, device):
    """
    Performs Macro-Average evaluation.
    Computes metrics per image, then averages them.
    """
    model.eval()
    
    metrics = {'iou': [], 'acc': [], 'f1': [], 'rec': [], 'prec': []}
    epsilon = 1e-6
    
    logger.info("Starting Macro-Average Evaluation...")

    with torch.no_grad():
        for _, (input_vl, input_ir, labels) in enumerate(loader):
            input_vl = input_vl.to(device)
            labels = labels.to(device)[:, 1, :, :].unsqueeze(1)
            
            outputs = model(input_vl)
            pre_final = outputs[0] if isinstance(outputs, tuple) else outputs
            
            batch_size = pre_final.shape[0]
            
            for i in range(batch_size):
                tp, fp, fn, tn = calculate_standard_confusion(
                    pre_final[i], labels[i], threshold=0.3
                )
                
                iou = tp / (tp + fp + fn + epsilon)
                acc = (tp + tn) / (tp + tn + fp + fn + epsilon)
                prec = tp / (tp + fp + epsilon)
                rec = tp / (tp + fn + epsilon)
                f1 = (2 * tp) / (2 * tp + fp + fn + epsilon)
                
                metrics['iou'].append(iou)
                metrics['acc'].append(acc)
                metrics['f1'].append(f1)
                metrics['rec'].append(rec)
                metrics['prec'].append(prec)

    results = {k: np.mean(v) for k, v in metrics.items()}
    
    return results['iou'], results['acc'], results['f1'], results['rec'], results['prec']


# ==============================================================================
# Main Execution
# ==============================================================================

def main(args):
    # 1. Device Setup
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu")
    logger.info(f"Running on device: {device}")

    # 2. Dataset Preparation
    test_vl_dir = os.path.join(args.data_root, 'vl')
    test_ir_dir = os.path.join(args.data_root, 'ir')
    test_gt_dir = os.path.join(args.data_root, 'gt')

    vl_list = [os.path.join(test_vl_dir, f) for f in get_sorted_files(test_vl_dir)]
    ir_list = [os.path.join(test_ir_dir, f) for f in get_sorted_files(test_ir_dir)]
    gt_list = [os.path.join(test_gt_dir, f) for f in get_sorted_files(test_gt_dir)]

    if not vl_list:
        logger.error("No dataset files found. Check your --data_root path.")
        return

    test_dataset = PL_dataset(vl_list, ir_list, gt_list, is_train=False)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 3. Model Initialization
    logger.info(f"Loading Model: {args.model_name}")
    try:
        model = eval(args.model_name)(n_class=1)
        model.to(device)
        if os.path.exists(args.weights):
            model.load_state_dict(torch.load(args.weights, map_location=device))
            logger.info(f"Weights loaded from: {args.weights}")
        else:
            logger.error(f"Weight file not found: {args.weights}")
            return
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        return

    # 4. Run Evaluation (Strict Mode)
    
    # A. Macro Evaluation (Standard)
    ma_iou, ma_acc, ma_f1, ma_rec, ma_prec = evaluate_macro_standard(
        model, test_loader, device
    )

    # B. Micro Evaluation (Standard)
    mi_iou, mi_acc, mi_f1, mi_rec, mi_prec = evaluate_micro_standard(
        model, test_loader, device
    )

    # 5. Report Results
    print("\n" + "="*60)
    print(f" FINAL PERFORMANCE REPORT (STRICT MODE - NO TOLERANCE) ")
    print("="*60)
    print(f"{'Metric':<20} | {'Macro (Standard)':<20} | {'Micro (Standard)':<20}")
    print("-" * 66)
    print(f"{'IoU':<20} | {ma_iou:.5f}{'':<12} | {mi_iou:.5f}")
    print(f"{'Accuracy':<20} | {ma_acc:.5f}{'':<12} | {mi_acc:.5f}")
    print(f"{'F1 Score':<20} | {ma_f1:.5f}{'':<12} | {mi_f1:.5f}")
    print(f"{'Recall':<20} | {ma_rec:.5f}{'':<12} | {mi_rec:.5f}")
    print(f"{'Precision':<20} | {ma_prec:.5f}{'':<12} | {mi_prec:.5f}")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Academic Evaluation Script: RGB-T (Standard)')
    parser.add_argument('--data_root', type=str, required=True, help='Root directory.')
    parser.add_argument('--weights', type=str, required=True, help='Path to .pth file.')
    parser.add_argument('--model_name', type=str, default='Your_model', help='Model class name.')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size.')
    parser.add_argument('--num_workers', type=int, default=4, help='CPU workers.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID.')

    args = parser.parse_args()
    main(args)