import os
import time
import argparse
import numpy as np 
import torch
from torch.utils.data import DataLoader

# TODO: Replace 'models' with your actual module containing the model architectures
# from models import Net 
from util.TTPLA_dataset import TTPLA_dataset

def get_batch_counts(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """
    Calculate True Positives (TP), False Positives (FP), False Negatives (FN), 
    and True Negatives (TN) for a batch of predictions.
    
    Args:
        pred (torch.Tensor): Model predictions.
        target (torch.Tensor): Ground truth labels.
        threshold (float): Binarization threshold for predictions.
        
    Returns:
        tuple: (tp, fp, fn, tn) counts as integers.
    """
    # Extract the target class dimension if the shape is [B, 2, H, W]
    if pred.dim() == 4 and pred.shape[1] == 2:
        pred = pred[:, 1, :, :]  
    # Squeeze the channel dimension if the shape is [B, 1, H, W]
    elif pred.dim() == 4 and pred.shape[1] == 1:
        pred = pred.squeeze(1)   
    
    # Process target tensor shapes similarly
    if target.dim() == 4 and target.shape[1] == 2:
        target = target[:, 1, :, :]  
    elif target.dim() == 4 and target.shape[1] == 1:
        target = target.squeeze(1)   

    # Apply sigmoid activation
    pred = torch.sigmoid(pred)
    
    # Binarize and flatten tensors for vectorized computation
    pred_mask = (pred > threshold).float().view(-1)
    target_mask = (target > 0).float().view(-1)

    # Safety check for shape mismatch
    if pred_mask.numel() != target_mask.numel():
        print(f"CRITICAL ERROR: Size mismatch between prediction and target!")
        print(f"Pred flat shape: {pred_mask.shape}, Target flat shape: {target_mask.shape}")
        return 0, 0, 0, 0

    # Calculate confusion matrix elements
    tp = (pred_mask * target_mask).sum().item()
    tn = ((1 - pred_mask) * (1 - target_mask)).sum().item()
    fp = (pred_mask * (1 - target_mask)).sum().item()
    fn = ((1 - pred_mask) * target_mask).sum().item()

    return tp, fp, fn, tn

def get_single_image_counts(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    """
    Calculate TP, FP, FN, TN for a single image tensor (C, H, W).
    """
    if pred.dim() == 3:
        if pred.shape[0] == 2:    # Dual channel output
            pred = pred[1, :, :]
        elif pred.shape[0] == 1:  # Single channel output
            pred = pred.squeeze(0)
            
    if target.dim() == 3:
        if target.shape[0] == 2:
            target = target[1, :, :]
        elif target.shape[0] == 1:
            target = target.squeeze(0)

    # Activate and binarize
    pred = torch.sigmoid(pred)
    pred_mask = (pred > threshold).float().view(-1)
    target_mask = (target > 0).float().view(-1)

    tp = (pred_mask * target_mask).sum().item()
    tn = ((1 - pred_mask) * (1 - target_mask)).sum().item()
    fp = (pred_mask * (1 - target_mask)).sum().item()
    fn = ((1 - pred_mask) * target_mask).sum().item()

    return tp, fp, fn, tn


def testing_Micro_without_exp(model, val_loader, threshold: float = 0.3):
    """
    Calculate global (Micro) metrics across the entire validation dataset.
    """
    model.eval()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    with torch.no_grad():
        for it, (input_vl, labels) in enumerate(val_loader):
            input_vl = input_vl.cuda()
            # Extract specific channel and keep dimensions [B, 1, H, W]
            labels = labels.cuda()[:, 1, :, :].unsqueeze(1)
            
            # Model inference (adjust unpacking based on your model's exact return signature)
            pre_final, pre_aux0, pre_aux1, pre_main, pre_fp, pre_fn = model(input_vl)
            
            # Calculate batch metrics
            tp, fp, fn, tn = get_batch_counts(pre_final, labels, threshold=threshold)
            
            # Accumulate to global counts
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    epsilon = 1e-6 # Epsilon to prevent division by zero

    # Calculate global metrics
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = (2 * precision * recall) / (precision + recall + epsilon)
    
    return acc, f1, recall, precision


def testing_Macro_without_exp(model, val_loader, threshold: float = 0.5):
    """
    Calculate average (Macro) Intersection over Union (IoU) per image.
    """
    model.eval()
    iou_list = []
    epsilon = 1e-6 

    with torch.no_grad():
        for it, (input_vl, labels) in enumerate(val_loader):
            input_vl = input_vl.cuda()
            labels = labels.cuda()[:, 1, :, :].unsqueeze(1) 
            
            # Model inference (ensure unpacking matches your model's return signature)
            outputs = model(input_vl)
            pre_final = outputs[0] if isinstance(outputs, tuple) else outputs
            
            batch_size = pre_final.shape[0]
            
            # Calculate IoU per image in the batch
            for i in range(batch_size):
                pred_i = pre_final[i] # Shape: (C, H, W)
                label_i = labels[i]   # Shape: (1, H, W)
                
                tp, fp, fn, tn = get_single_image_counts(pred_i, label_i, threshold=threshold)
                
                # Intersection over Union formula: TP / (TP + FP + FN)
                iou = tp / (tp + fp + fn + epsilon)
                iou_list.append(iou)
                
    avg_iou = np.mean(iou_list)
    return avg_iou


def main(args):
    # Set up dataset paths
    vl_test_list = os.listdir(args.test_vl_dir)
    gt_test_list = os.listdir(args.test_gt_dir)
    
    absolute_vl_test_list = [os.path.abspath(os.path.join(args.test_vl_dir, f)) for f in vl_test_list]
    absolute_gt_test_list = [os.path.abspath(os.path.join(args.test_gt_dir, f)) for f in gt_test_list]

    # Initialize Dataset and DataLoader
    testing_dataset = TTPLA_dataset(absolute_vl_test_list, absolute_gt_test_list, is_train=False)
    
    testing_loader = DataLoader(
        dataset     = testing_dataset,
        batch_size  = args.batch_size,
        shuffle     = False,
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    
    print(f"[*] Dataset loaded. Total testing samples: {len(testing_dataset)}")

    # Initialize Model
    # Note: It is safer to import the model directly rather than using eval()
    # e.g., model = Net(n_class=args.n_class)
    model = eval(args.model_name)(n_class=args.n_class) 
    
    if args.gpu >= 0: 
        model.cuda(args.gpu)
    
    # Load pretrained weights
    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Weight file not found at: {args.weight_path}")
        
    print(f"[*] Loading weights from {args.weight_path}...")
    model.load_state_dict(torch.load(args.weight_path))
    model.cuda()

    # Evaluation
    print("[*] Starting evaluation...")
    start_time = time.time()
    
    acc, f1, recall, precision = testing_Micro_without_exp(model, testing_loader, threshold=0.3)  
    macro_iou = testing_Macro_without_exp(model, testing_loader, threshold=0.5) 
    
    end_time = time.time()
    
    # Print Results
    print("-" * 50)
    print("Evaluation Results:")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Macro IoU : {macro_iou:.4f}")
    print(f"Time Taken: {end_time - start_time:.2f} seconds")
    print("-" * 50)
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Testing Script for Segmentation')
    
    # Model configs
    parser.add_argument('--model_name',  '-M',  type=str, default='Net', help='Name of the model class')
    parser.add_argument('--n_class',     '-C',  type=int, default=1, help='Number of output classes')
    
    # Dataset and paths
    parser.add_argument('--test_vl_dir', type=str, required=True, help='Path to the test images directory')
    parser.add_argument('--test_gt_dir', type=str, required=True, help='Path to the test ground truth directory')
    parser.add_argument('--weight_path', type=str, required=True, help='Path to the saved model weights (.pth)')
    
    # Dataloader configs
    parser.add_argument('--batch_size',  '-B',  type=int, default=4, help='Batch size for testing') 
    parser.add_argument('--num_workers', '-j',  type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--gpu',         '-G',  type=int, default=0, help='GPU ID to use')
    
    args = parser.parse_args()
    
    main(args)