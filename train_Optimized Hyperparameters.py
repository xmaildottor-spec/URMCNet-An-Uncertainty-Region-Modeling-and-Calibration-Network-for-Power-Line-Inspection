# coding:utf-8 
import os
import argparse
import time
import copy
import logging
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# External dependencies
from util.PL_dataset import PL_dataset
from util.util import calc_loss, print_metrics
from model import Your_Model

# --- 1. Learning Rate Scheduler (Warmup + Polynomial Decay) ---

class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Custom Scheduler: Linear Warmup followed by Polynomial Decay.
    Formula for decay: lr = initial_lr * (1 - iter / max_iter) ^ power
    """
    def __init__(self, optimizer, target_lr, max_iters, warmup_iters, power=0.9, last_epoch=-1):
        self.target_lr = target_lr
        self.max_iters = max_iters
        self.warmup_iters = warmup_iters
        self.power = power
        super(WarmupPolyLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        current_iter = self.last_epoch
        if current_iter < self.warmup_iters:
            # Linear Warmup phase
            factor = current_iter / self.warmup_iters
        else:
            # Polynomial Decay phase
            factor = (1 - (current_iter - self.warmup_iters) / (self.max_iters - self.warmup_iters)) ** self.power
        
        return [self.target_lr * max(factor, 1e-6) for _ in self.base_lrs]

# --- 2. Trainer Engine ---

class Trainer:
    def __init__(self, args, model, train_loader, val_loader):
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Hyperparameters: AdamW with specific weight decay
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=args.weight_decay
        )
        
        # Scheduler calculations (Step-based)
        total_steps = len(train_loader) * args.epoch_max
        warmup_steps = len(train_loader) * args.warmup_epochs
        self.scheduler = WarmupPolyLR(
            self.optimizer, 
            target_lr=args.lr, 
            max_iters=total_steps, 
            warmup_iters=warmup_steps, 
            power=0.9
        )

        self.start_epoch = 1
        self.best_loss = float('inf')
        self.history = {'train': [], 'val': []}

        # Handle Breakpoint / Resume Strategy
        if args.resume:
            self._resume_checkpoint()

    def _resume_checkpoint(self):
        ckpt_path = os.path.join(self.args.save_dir, "checkpoint_latest.pth")
        if os.path.exists(ckpt_path):
            logging.info(f"==> Loading checkpoint: {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_loss = checkpoint['best_loss']
        else:
            logging.warning("==> No checkpoint found at specified path. Starting from scratch.")

    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
        }
        # Save latest for resume
        latest_path = os.path.join(self.args.save_dir, "checkpoint_latest.pth")
        torch.save(state, latest_path)
        
        # Save best model separately
        if is_best:
            best_path = os.path.join(self.args.save_dir, f"{self.args.model_name}_best_seed{self.args.rand_seed}.pth")
            torch.save(self.model.state_dict(), best_path)
            logging.info(f"New best model achieved at epoch {epoch}")

    def run_epoch(self, epoch, phase='train'):
        if phase == 'train':
            self.model.train()
            loader = self.train_loader
        else:
            self.model.eval()
            loader = self.val_loader

        metrics = defaultdict(float)
        pbar = tqdm(enumerate(loader), total=len(loader), desc=f"{phase.capitalize()} Epoch {epoch}")

        for i, (input_vl, input_ir, labels) in pbar:
            input_vl = input_vl.to(self.device)
            input_ir = input_ir.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(phase == 'train'):
                outputs = self.model(input_vl, input_ir)
                # outputs = self.model(input_vl)
                loss, metrics = calc_loss(outputs, labels, metrics)

                if phase == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step() # Iteration-level update

            # Update progress bar
            if phase == 'train':
                pbar.set_postfix(loss=loss.item(), lr=self.optimizer.param_groups[0]['lr'])

        # Calculate epoch averages
        avg_metrics = {k: v / len(loader) for k, v in metrics.items()}
        return avg_metrics

    def fit(self):
        for epoch in range(self.start_epoch, self.args.epoch_max + 1):
            train_m = self.run_epoch(epoch, 'train')
            val_m = self.run_epoch(epoch, 'val')
            
            self.history['train'].append([train_m['bce'], train_m['dice'], train_m['loss']])
            self.history['val'].append([val_m['bce'], val_m['dice'], val_m['loss']])

            logging.info(f"Epoch {epoch} Summary: Train Loss: {train_m['loss']:.4f} | Val Loss: {val_m['loss']:.4f}")

            # Best model selection
            is_best = val_m['loss'] < self.best_loss
            if is_best:
                self.best_loss = val_m['loss']
            
            self.save_checkpoint(epoch, is_best)

        self._export_logs()

    def _export_logs(self):
        df1 = pd.DataFrame(self.history['train'], columns=['bce', 'dice', 'loss'])
        df2 = pd.DataFrame(self.history['val'], columns=['bce', 'dice', 'loss'])
        path = os.path.join(self.args.save_dir, f"training_log_seed{self.args.rand_seed}.xlsx")
        with pd.ExcelWriter(path) as writer:
            df1.to_excel(writer, sheet_name='train')
            df2.to_excel(writer, sheet_name='val')
        logging.info(f"Excel logs exported to {path}")

# --- 3. Entry Point ---

def main():
    parser = argparse.ArgumentParser(description='Multi-Modal Training Script')
    parser.add_argument('--model_name',  '-M',  type=str, default='Your_Model')
    parser.add_argument('--batch_size',  '-B',  type=int, default=8)
    parser.add_argument('--epoch_max' ,  '-E',  type=int, default=200)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--lr',          '-LR', type=float, default=3e-4)
    parser.add_argument('--weight_decay', '-WD', type=float, default=0.005)
    parser.add_argument('--gpu',         '-G',  type=int, default=0)
    parser.add_argument('--num_workers', '-j',  type=int, default=4)
    parser.add_argument('--rand_seed',   '-RS', type=int, default=1)
    parser.add_argument('--resume',      '-R',  action='store_true', help='Resume from latest checkpoint')
    args = parser.parse_args()

    # Path Setup
    # Note: Replace 'model_dir' with your actual root save directory
    model_root_dir = './checkpoints' 
    args.save_dir = os.path.join(model_root_dir, args.model_name)
    os.makedirs(args.save_dir, exist_ok=True)

    # Logging Setup
    log_format = '%(asctime)s %(levelname)s: %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format,
                        handlers=[logging.FileHandler(os.path.join(args.save_dir, "train.log")),
                                  logging.StreamHandler()])

    # Dataset Loading
    # TODO: Replace the following lists with actual data loading logic
    vl_list, ir_list, gt_list = [], [], [] 
    
    train_dataset = PL_dataset(vl_list, ir_list, gt_list, is_train=True)
    val_dataset   = PL_dataset(vl_list, ir_list, gt_list, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            num_workers=args.num_workers, pin_memory=True)

    # Model Initialization
    # Using globals to dynamically instantiate the class by name
    if args.model_name in globals():
        model = globals()[args.model_name](n_class=2)
    else:
        raise ValueError(f"Model {args.model_name} not recognized.")

    # Start Training
    trainer = Trainer(args, model, train_loader, val_loader)
    trainer.fit()

if __name__ == '__main__':
    main()