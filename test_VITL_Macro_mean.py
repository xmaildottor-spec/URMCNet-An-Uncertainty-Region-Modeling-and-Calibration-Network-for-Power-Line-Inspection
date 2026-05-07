import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from util.VITLD_dataset import PL_dataset

def get_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple:
    """
    Processes dual-channel prediction and target tensors to calculate TP, FP, FN, TN.
    
    Args:
        pred (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth labels.
        threshold (float): Binarization threshold for predictions.
        
    Returns:
        tuple: (True Positives, False Positives, False Negatives, True Negatives)
    """
    # 1. Process prediction tensor
    if pred.dim() == 4 and pred.shape[1] == 2:
        pred = pred[:, 1, :, :]  
    elif pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)   
    
    # 2. Process target tensor
    if target.dim() == 4 and target.shape[1] == 2:
        target = target[:, 1, :, :]  
    elif target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)   

    # 3. Activation and Binarization
    pred = torch.sigmoid(pred)
    pred_mask = (pred > threshold).float().view(-1)
    target_mask = (target > 0).float().view(-1)

    if pred_mask.numel() != target_mask.numel():
        raise ValueError(f"Size mismatch: Pred shape {pred_mask.shape} vs Target shape {target_mask.shape}")

    # 4. Calculate metrics
    tp = (pred_mask * target_mask).sum().item()
    tn = ((1 - pred_mask) * (1 - target_mask)).sum().item()
    fp = (pred_mask * (1 - target_mask)).sum().item()
    fn = ((1 - pred_mask) * target_mask).sum().item()

    return tp, fp, fn, tn


def evaluate_global_metrics(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, threshold: float = 0.3) -> tuple:
    """
    Evaluates the model across the entire dataset to compute global (micro) class-macro averaged metrics.
    
    Args:
        model (torch.nn.Module): The segmentation model.
        val_loader (DataLoader): DataLoader for the validation/test dataset.
        device (torch.device): Compute device (CPU or GPU).
        threshold (float): Binarization threshold.
        
    Returns:
        tuple: (mIoU, Accuracy, mF1, mRecall, mPrecision)
    """
    model.eval()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    with torch.no_grad():
        for input_vl, input_ir, labels in val_loader:
            input_vl = input_vl.to(device)
            # Ensure label is correctly shaped for comparison (B, 1, H, W)
            labels = labels.to(device)[:, 1, :, :].unsqueeze(1)
            
            # Forward pass
            pre_final, _, _, _, _, _ = model(input_vl)
            
            # Calculate counts
            tp, fp, fn, tn = get_confusion_matrix(pre_final, labels, threshold=threshold)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    epsilon = 1e-6

    # 1. Foreground Metrics (Class 1)
    fg_iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
    fg_prec = total_tp / (total_tp + total_fp + epsilon)
    fg_rec = total_tp / (total_tp + total_fn + epsilon)
    fg_f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + epsilon)

    # 2. Background Metrics (Class 0: TN acts as TP, FN as FP, FP as FN)
    bg_iou = total_tn / (total_tn + total_fn + total_fp + epsilon)
    bg_prec = total_tn / (total_tn + total_fn + epsilon)
    bg_rec = total_tn / (total_tn + total_fp + epsilon)
    bg_f1 = (2 * total_tn) / (2 * total_tn + total_fn + total_fp + epsilon)

    # 3. Class-Macro Averaging
    m_iou = (fg_iou + bg_iou) / 2.0
    m_prec = (fg_prec + bg_prec) / 2.0
    m_rec = (fg_rec + bg_rec) / 2.0
    m_f1 = (fg_f1 + bg_f1) / 2.0
    
    # 4. Global Pixel Accuracy
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)

    return m_iou, acc, m_f1, m_rec, m_prec


def main(args):
    # Setup device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    print(f"[*] Initializing evaluation on {device}...")

    # Validate dataset paths
    if not all(os.path.exists(p) for p in [args.vl_dir, args.ir_dir, args.gt_dir]):
        raise FileNotFoundError("One or more dataset directories do not exist. Please check your paths.")

    # Prepare dataset lists
    vl_list = [os.path.abspath(os.path.join(args.vl_dir, f)) for f in os.listdir(args.vl_dir)]
    ir_list = [os.path.abspath(os.path.join(args.ir_dir, f)) for f in os.listdir(args.ir_dir)]
    gt_list = [os.path.abspath(os.path.join(args.gt_dir, f)) for f in os.listdir(args.gt_dir)]

    # Initialize Dataset and DataLoader
    testing_dataset = PL_dataset(vl_list, ir_list, gt_list, is_train=False)
    testing_loader = DataLoader(
        dataset=testing_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Initialize Model
    model = eval(args.model_name)(n_class=args.n_class)
    
    # Load Weights
    if not os.path.exists(args.weights):
        raise FileNotFoundError(f"Model weights not found at: {args.weights}")
    
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)

    # Run Evaluation
    Macro_iou, _, Macro_f1, Macro_recall, Macro_precision = evaluate_global_metrics(
        model=model, 
        val_loader=testing_loader, 
        device=device, 
        threshold=0.3
    )  
       
    # Output Results
    print('\n=============== Class-Macro Averaged Metrics ===============')
    print(f'| Macro-mIoU      : {Macro_iou:.4f}')
    print(f'| Macro-mF1       : {Macro_f1:.4f}')
    print(f'| Macro-mRecall   : {Macro_recall:.4f}')
    print(f'| Macro-mPrecision: {Macro_precision:.4f}')
    print('------------------------------------------------------------\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Image Segmentation Model')
    
    # Model Configuration
    parser.add_argument('--model_name', '-M', type=str, default='Net', help='Name of the model class to evaluate')
    parser.add_argument('--weights', '-W', type=str, required=True, help='Path to the pre-trained model weights (.pth)')
    parser.add_argument('--n_class', type=int, default=1, help='Number of target classes')
    
    # Dataset Paths
    parser.add_argument('--vl_dir', type=str, required=True, help='Directory containing Visible Light (VL) images')
    parser.add_argument('--ir_dir', type=str, required=True, help='Directory containing Infrared (IR) images')
    parser.add_argument('--gt_dir', type=str, required=True, help='Directory containing Ground Truth (GT) labels')
    
    # Hardware & DataLoader Configuration
    parser.add_argument('--batch_size', '-B', type=int, default=4, help='Batch size for evaluation')
    parser.add_argument('--num_workers', '-j', type=int, default=0, help='Number of DataLoader workers')
    parser.add_argument('--gpu', '-G', type=int, default=0, help='GPU ID to use. Set to -1 for CPU.')
    
    args = parser.parse_args()
    
    main(args)