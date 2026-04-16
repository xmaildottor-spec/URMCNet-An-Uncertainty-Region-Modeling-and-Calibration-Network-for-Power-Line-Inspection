import os
import time
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

from util.TTPLA_dataset import TTPLA_dataset_WithExpansion
from URMCNet_res_FPFN import TFNet 

def get_sorted_files(directory):
    """
    Fetch and sort valid image files from a specified directory.
    Ensures consistent order across predictions and different ground truth variants.
    """
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in valid_extensions]
    return sorted(files)

def get_single_image_counts_tolerant(pred, target_original, target_expanded, threshold=0.3):
    """
    Calculate TP, FP, FN, TN for a single image under the tolerance mechanism.
    
    Tolerance rules:
    - TP: Predicted positive AND falls within the expanded GT region.
    - FP: Predicted positive BUT falls completely outside the expanded GT region.
    - FN: Predicted negative BUT belongs to the original core GT target.
    - TN: Remaining pixels to ensure conservation.
    """
    # 1. Dimension processing: adapt to 1-channel output structure (H, W)
    if pred.dim() == 3: 
        pred = pred.squeeze(0)
    if target_original.dim() == 3: 
        target_original = target_original.squeeze(0)
    if target_expanded.dim() == 3: 
        target_expanded = target_expanded.squeeze(0)

    # 2. Generate boolean masks
    pred = torch.sigmoid(pred)
    pred_mask = (pred > threshold)          # Positive predictions
    gt_core_mask = (target_original > 0)    # Original core targets
    gt_exp_mask = (target_expanded > 0)     # Expanded tolerance regions

    # 3. Calculate tolerant metrics
    tp_mask = pred_mask & gt_exp_mask 
    tp = tp_mask.sum().item()

    fp_mask = pred_mask & (~gt_exp_mask)
    fp = fp_mask.sum().item()

    fn_mask = (~pred_mask) & gt_core_mask
    fn = fn_mask.sum().item()

    tn = pred_mask.numel() - tp - fp - fn

    return tp, fp, fn, tn

def testing_Hybrid_WithExpansion(model, val_loader):
    """
    Evaluates the model using a hybrid metric approach:
    Macro IoU (averaged per image) and Micro ACC/F1/Recall/Precision (accumulated globally).
    """
    model.eval()
    
    # List to record IoU for each image (for Macro computation)
    iou_list = []
    
    # Global accumulators (for Micro computation)
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    epsilon = 1e-6 
    
    print("Start Hybrid Testing (Macro IoU + Micro others) with Tolerance Mechanism...")

    with torch.no_grad():
        for it, (input_vl, labels, labels_exp) in enumerate(val_loader):
            
            input_vl = input_vl.cuda()
            
            # Label processing: extract foreground channel and adjust dimensions to [B, 1, H, W]
            labels = labels.cuda()[:, 1, :, :].unsqueeze(1)      
            labels_exp = labels_exp.cuda()                       
            
            # Model inference
            pre_final, pre_aux0, pre_aux1, pre_main, pre_fp, pre_fn = model(input_vl)
            
            batch_size = pre_final.shape[0]
            
            # Iterate through each image in the current batch
            for i in range(batch_size):
                pred_i = pre_final[i]      
                label_i = labels[i]        
                label_exp_i = labels_exp[i]
                
                # Calculate metric counts for the current image
                tp, fp, fn, tn = get_single_image_counts_tolerant(pred_i, label_i, label_exp_i, threshold=0.3)
                
                # 1. Calculate single-image IoU and append to list (Macro approach)
                iou = tp / (tp + fp + fn + epsilon)
                iou_list.append(iou)
                
                # 2. Accumulate to global counters (Micro approach)
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn

    # === Compute Macro Metric ===
    macro_iou = np.mean(iou_list)
    
    # === Compute Micro Metrics ===
    micro_acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
    micro_precision = total_tp / (total_tp + total_fp + epsilon)
    micro_recall = total_tp / (total_tp + total_fn + epsilon)
    micro_f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + epsilon)

    print("Testing Finished (Hybrid Mode).")
    return macro_iou, micro_acc, micro_f1, micro_recall, micro_precision

def main(args):
    # Retrieve file lists
    vl_test_list = get_sorted_files(args.vl_dir)
    gt_test_list = get_sorted_files(args.gt_dir)
    exp_test_list = get_sorted_files(args.exp_dir) 

    # Validate dataset consistency
    assert len(vl_test_list) == len(exp_test_list), "Error: The number of expanded labels does not match the test set size!"

    # Convert to absolute paths
    absolute_vl_test_list = [os.path.join(args.vl_dir, f) for f in vl_test_list]
    absolute_gt_test_list = [os.path.join(args.gt_dir, f) for f in gt_test_list]
    absolute_exp_test_list = [os.path.join(args.exp_dir, f) for f in exp_test_list] 

    # Initialize Dataset and DataLoader
    testing_dataset = TTPLA_dataset_WithExpansion(
        absolute_vl_test_list, 
        absolute_gt_test_list, 
        absolute_exp_test_list, 
        is_train=False
    )
    
    testing_loader = DataLoader(
        dataset     = testing_dataset,
        batch_size  = args.batch_size,
        shuffle     = False, # Shuffle strictly prohibited during testing to maintain order
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
    
    # Initialize Model
    n_class = 1
    model = eval(args.model_name)(n_class=n_class)
    
    if args.gpu >= 0: 
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    
    # Load weights
    model.load_state_dict(torch.load(args.weight_path, map_location=f'cuda:{args.gpu}'))
    model.eval()
   
    # Execute evaluation
    Macro_iou, Micro_acc, Micro_f1, Micro_recall, Micro_precision = testing_Hybrid_WithExpansion(model, testing_loader) 

    print('-' * 45)
    print('| Macro (Tolerant) IoU       : {:.4f}'.format(Macro_iou))
    print('| Micro (Tolerant) Accuracy  : {:.4f}'.format(Micro_acc))
    print('| Micro (Tolerant) F1-Score  : {:.4f}'.format(Micro_f1))
    print('| Micro (Tolerant) Recall    : {:.4f}'.format(Micro_recall))
    print('| Micro (Tolerant) Precision : {:.4f}'.format(Micro_precision))
    print('-' * 45)
            
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hybrid Metric Evaluation with Tolerance Mechanism (TTPLA)')
    
    # Dataset path arguments
    parser.add_argument('--vl_dir', type=str, required=True, help='Path to visible light test images')
    parser.add_argument('--gt_dir', type=str, required=True, help='Path to standard ground truth labels')
    parser.add_argument('--exp_dir', type=str, required=True, help='Path to expanded ground truth labels')
    
    # Model arguments
    parser.add_argument('--model_name', '-M', type=str, default='TFNet', help='Name of the model architecture')
    parser.add_argument('--weight_path', '-W', type=str, required=True, help='Path to the pre-trained model weights (.pth)')
    
    # System arguments
    parser.add_argument('--batch_size', '-B', type=int, default=4, help='Batch size for testing') 
    parser.add_argument('--num_workers', '-j', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--gpu', '-G', type=int, default=0, help='GPU device ID to use')
    
    args = parser.parse_args()
   
    main(args)