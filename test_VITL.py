import torch
import numpy as np 
from setproctitle import setproctitle
import os
import cv2 
setproctitle('lyc')
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  
import argparse, time
from torch.utils.data import DataLoader
from util.VITLD_dataset import PL_dataset, PL_dataset_WithExpansion
# from URMCNet_res_FPFN import TFNet 
from URMCNet_star_FPFN import TFNet 


times = time.time()
local_time = time.localtime(times)

n_class = 1

#Path Config
test_vl_dir = r'-------------------------------------'
test_ir_dir = r'-------------------------------------'
test_gt_dir = r'-------------------------------------'
test_exp_dir = r'-------------------------------------'


def get_sorted_files(directory):
    valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    files = [f for f in os.listdir(directory) if os.path.splitext(f)[1].lower() in valid_extensions]
    return sorted(files)

vl_test_list = get_sorted_files(test_vl_dir)
ir_test_list = get_sorted_files(test_ir_dir)
gt_test_list = get_sorted_files(test_gt_dir)
exp_test_list = get_sorted_files(test_exp_dir) # 获取膨胀文件列表

assert len(vl_test_list) == len(exp_test_list), "错误：膨胀标签文件夹中的图片数量与测试集不一致！"


absolute_vl_test_list = [os.path.join(test_vl_dir, f) for f in vl_test_list]
absolute_ir_test_list = [os.path.join(test_ir_dir, f) for f in ir_test_list]
absolute_gt_test_list = [os.path.join(test_gt_dir, f) for f in gt_test_list]
absolute_exp_test_list = [os.path.join(test_exp_dir, f) for f in exp_test_list] # 膨胀标签绝对路径

testing_dataset = PL_dataset_WithExpansion(absolute_vl_test_list, absolute_ir_test_list, absolute_gt_test_list, absolute_exp_test_list, is_train=False)

def get_batch_counts_tolerant(pred, target_original, target_expanded, threshold=0.5):

    if pred.dim() == 4: pred = pred.squeeze(1)
    if target_original.dim() == 4: target_original = target_original.squeeze(1)
    if target_expanded.dim() == 4: target_expanded = target_expanded.squeeze(1)

    pred = torch.sigmoid(pred)
    pred_mask = (pred > threshold)          
    gt_core_mask = (target_original > 0)    
    gt_exp_mask = (target_expanded > 0)     

    tp_mask = pred_mask & gt_exp_mask 
    tp = tp_mask.sum().item()

    fp_mask = pred_mask & (~gt_exp_mask)
    fp = fp_mask.sum().item()

    fn_mask = (~pred_mask) & gt_core_mask
    fn = fn_mask.sum().item()

    tn = pred_mask.numel() - tp - fp - fn

    return tp, fp, fn, tn

def testing(model, val_loader):
    model.eval()
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    
    print("Start Testing with Tolerance Mechanism (Auto-Loading)...")

    with torch.no_grad():

        for it, (input_vl, input_ir, labels, labels_exp) in enumerate(val_loader):
            
            input_vl = input_vl.cuda()
            
            labels = labels.cuda()[:, 1, :, :].unsqueeze(1) 
            
            labels_exp = labels_exp.cuda()
            
            pre_final, pre_aux0, pre_aux1, pre_main, pre_fp, pre_fn = model(input_vl)
            
            tp, fp, fn, tn = get_batch_counts_tolerant(pre_final, labels, labels_exp, threshold=0.3)
            
            total_tp += tp
            total_fp += fp
            total_fn += fn
            total_tn += tn

    epsilon = 1e-6
    iou = total_tp / (total_tp + total_fp + total_fn + epsilon)
    acc = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + epsilon)
    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = (2 * total_tp) / (2 * total_tp + total_fp + total_fn + epsilon)

    print(f"Testing Finished (Tolerant Mode).")
    print(f"IoU: {iou:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}, Acc: {acc:.4f}")
    
    return iou, acc, f1, recall, precision


def main():
    
    testing_loader  = DataLoader(
        dataset     = testing_dataset,
        batch_size  = args.batch_size,
        shuffle     = False, 
        num_workers = args.num_workers,
        pin_memory  = True,
        drop_last   = False
    )
   
   
    iou, acc, f1, recall, precision = testing(model, testing_loader)  
   
    print('| Micro iou is : %s' % iou)
    print('| Micro acc is : %s' % acc)
    print('| Micro f1 is : %s' % f1)
    print('| Micro recall is : %s' % recall)
    print('| Micro precision is : %s' % precision)
          
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Test with pytorch')
    parser.add_argument('--model_name',  '-M',  type=str, default='TFNet')
    parser.add_argument('--batch_size',  '-B',  type=int, default=4) 
    parser.add_argument('--num_workers', '-j',  type=int, default=0) 
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    args = parser.parse_args()
    
    model = eval(args.model_name)(n_class=n_class)
    if args.gpu >= 0: model.cuda(args.gpu)
    
    model.load_state_dict(torch.load(r'----------------------------------------'))

    model.cuda()
   
    main()