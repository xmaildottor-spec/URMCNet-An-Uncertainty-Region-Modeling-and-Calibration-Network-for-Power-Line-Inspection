# coding:utf-8
import os
import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
import random 
import numpy as np  

# ================== Random Seed Setup ==================
def setup_seed(seed):
    """Fix random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Disable cuDNN benchmark and enable deterministic for strict reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================== Dynamic Weight Function ==================
def get_decay_weight(epoch, max_epochs, start_val, end_val, power=0.9):
    """
    Calculate the loss weight for the current epoch.
    :param power: Decay exponent. 1.0 is linear decay, 0.9 is standard Poly decay.
    """
    if max_epochs == 0:  # Avoid division by zero
        return start_val
    current_epoch = min(epoch, max_epochs)
    decay_factor = (1 - current_epoch / max_epochs) ** power
    current_weight = (start_val - end_val) * decay_factor + end_val
    return current_weight

# ================== Poly LR ==================
def poly_lr(optimizer, cur_iter, max_iter, base_lr, power=0.9):
    """Apply standard Poly learning rate decay."""
    lr = base_lr * (1 - cur_iter / max_iter) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

# ================== Training Routine ==================
def train_one_epoch(model, loader, optimizer, epoch, max_iter, args):
    model.train()
  
    # Optional: Dynamic weights (currently commented out in original logic)
    # w_aux0 = get_decay_weight(epoch, args.epoch_max, start_val=0.8, end_val=0.1, power=0.9)
    # w_aux1 = get_decay_weight(epoch, args.epoch_max, start_val=0.6, end_val=0.1, power=0.9)
    # print(f"| Epoch {epoch} Weights | Aux0: {w_aux0:.4f} | Aux1: {w_aux1:.4f}")

    for it, (input_vl, input_ir, labels, labels_fn, labels_fp) in enumerate(loader):
        
        input_vl = input_vl.cuda(args.gpu)
        input_ir = input_ir.cuda(args.gpu)
        
        labels = labels.cuda(args.gpu)[:, 1, :, :].unsqueeze(1)
        labels_fn = labels_fn.cuda(args.gpu)[:, 1, :, :].unsqueeze(1)
        labels_fp = labels_fp.cuda(args.gpu)[:, 1, :, :].unsqueeze(1)

        pre_final, pre_aux0, pre_aux1, pre_main, pre_fp, pre_fn = model(input_vl)

        loss_final = calc_loss_one_channel(pre_final, labels)
        loss_aux0 = calc_loss_one_channel(pre_aux0, labels)
        loss_aux1 = calc_loss_one_channel(pre_aux1, labels)
        loss_main = calc_loss_one_channel(pre_main, labels)
        loss_fn = calc_loss_one_channel(pre_fn, labels_fn)
        loss_fp = calc_loss_one_channel(pre_fp, labels_fp)

        # Static weighted sum computation
        loss = loss_final + 0.25 * loss_fn + 0.25 * loss_fp + 0.8 * loss_aux0 + 0.6 * loss_aux1 + loss_main   

        # Alternative: Weighted sum using dynamic weights
        # loss = loss_final + 0.25 * loss_fn + 0.25 * loss_fp + w_aux0 * loss_aux0 + w_aux1 * loss_aux1 + loss_main 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cur_iter = epoch * loader.n_iter + it
        poly_lr(optimizer, cur_iter, max_iter, args.lr_start)

        if it % 20 == 0:
            print(f'| Epoch {epoch} | Iter {it}/{loader.n_iter} | Loss {loss.item():.4f}')

# ================== Main Process ==================
def main(args):
    torch.cuda.set_device(args.gpu)

    model = Net(n_class=args.n_class).cuda(args.gpu)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr_start,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )

    # ------------------ Data Paths Configuration ------------------
    # Get list of files in the directories
    vl_tra_list = os.listdir(args.train_vl_dir)
    ir_tra_list = os.listdir(args.train_ir_dir)
    gt_tra_list = os.listdir(args.train_gt_dir)

    fngt_tra_list = os.listdir(args.train_fngt_dir)
    fpgt_tra_list = os.listdir(args.train_fpgt_dir)

    vl_test_list = os.listdir(args.test_vl_dir)
    ir_test_list = os.listdir(args.test_ir_dir)
    gt_test_list = os.listdir(args.test_gt_dir)

    # Convert relative paths to absolute paths dynamically
    absolute_vl_tra_list = [os.path.abspath(os.path.join(args.train_vl_dir, f)) for f in vl_tra_list]
    absolute_ir_tra_list = [os.path.abspath(os.path.join(args.train_ir_dir, f)) for f in ir_tra_list]
    absolute_gt_tra_list = [os.path.abspath(os.path.join(args.train_gt_dir, f)) for f in gt_tra_list]

    absolute_fngt_tra_list = [os.path.abspath(os.path.join(args.train_fngt_dir, f)) for f in fngt_tra_list]
    absolute_fpgt_tra_list = [os.path.abspath(os.path.join(args.train_fpgt_dir, f)) for f in fpgt_tra_list]

    absolute_vl_test_list = [os.path.abspath(os.path.join(args.test_vl_dir, f)) for f in vl_test_list]
    absolute_ir_test_list = [os.path.abspath(os.path.join(args.test_ir_dir, f)) for f in ir_test_list]
    absolute_gt_test_list = [os.path.abspath(os.path.join(args.test_gt_dir, f)) for f in gt_test_list]

    train_dataset = PL_fpfn_dataset(
        absolute_vl_tra_list, absolute_ir_tra_list, absolute_gt_tra_list,
        absolute_fngt_tra_list, absolute_fpgt_tra_list, is_train=True
    )
    test_dataset = PL_dataset(
        absolute_vl_test_list, absolute_ir_test_list, absolute_gt_test_list, is_train=False
    )

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
    test_loader  = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=args.num_workers)
  
    train_loader.n_iter = len(train_loader)
    test_loader.n_iter  = len(test_loader)

    max_iter = args.epoch_max * train_loader.n_iter

    # Ensure output directories exist
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    
    for epoch in range(args.epoch_max):
        train_one_epoch(model, train_loader, optimizer, epoch, max_iter, args)

        if epoch > 150 or epoch % 10 == 0 or epoch == 0:
            iou, acc, f1, recall, precision = testing_Macro(model, test_loader)
          
            if iou > 0.650:
                save_path = os.path.join(args.save_dir, f'{args.target}_{epoch}.pth')
                torch.save(model.state_dict(), save_path)
                print(f'Model saved to: {save_path}')

            print('Evaluation OK!')
            with open(args.log_path, 'a') as log_file:
                log_file.write(
                    f'epoch: {epoch}   iou: {iou:.3f}  acc: {acc:.3f}  '
                    f'f1: {f1:.3f}  recall: {recall:.3f}  precision: {precision:.3f}\n'
                )
            print(f'Metrics logged to: {args.log_path}')

    # ================== Record Training End Time ==================
    end_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    
    with open(args.log_path, 'a') as log_file:
        log_file.write(f'--------------------- TRAINING FINISHED ----------------------- {end_time_str}\n')
    print(f"All training finished at {end_time_str}")

# ================== Entry Point ==================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    
    # Basic Configurations
    parser.add_argument('--n_class', type=int, default=1, help='Number of classes')
    parser.add_argument('--lr_start', type=float, default=3e-4, help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.005, help='Optimizer weight decay')
    parser.add_argument('--target', type=str, default='------', help='Target model name')
    
    # Dataloader & Environment Configurations
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
    parser.add_argument('--epoch_max', type=int, default=300, help='Maximum number of epochs')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=4200, help='Random seed for reproducibility')
    
    # I/O Paths (Refactored for public sharing)
    parser.add_argument('--data_root', type=str, default='------', help='Root directory for dataset')
    parser.add_argument('--save_dir', type=str, default='------', help='Directory to save model weights')
    parser.add_argument('--log_path', type=str, default='------', help='Path to save training logs')

    args = parser.parse_args()
    
    # Dynamically build data paths based on data_root
    args.train_vl_dir = os.path.join(args.data_root, '------')
    args.train_ir_dir = os.path.join(args.data_root, '------')
    args.train_gt_dir = os.path.join(args.data_root, '------')
    args.train_fngt_dir = os.path.join(args.data_root, '------')
    args.train_fpgt_dir = os.path.join(args.data_root, '------')

    args.test_vl_dir = os.path.join(args.data_root, '------')
    args.test_ir_dir = os.path.join(args.data_root, '------')
    args.test_gt_dir = os.path.join(args.data_root, '------')

    setup_seed(args.seed)

    # Initialize log file
    os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
    with open(args.log_path, 'a') as log_file:
        log_file.write(f'--------------------- {args.target} ----------------------- {start_time_str}\n')

    main(args)