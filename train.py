import os 
import time
import torch

import torch.optim as optim
# from tqdm import tqdm

# import torchvision
import argparse
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

import config
from MultiJODDataset import *
from MultiJODModel import *
from dataset import CUDAPrefetcher
from Meter import AverageMeter, ProgressMeter


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

def load_dataset():
    train_patch_info_csv = f'{config.train_image_dir}/patch_info.csv'
    train_video_info_csv = f'{config.train_image_dir}/video_info.csv'
    val_patch_info_csv = f'{config.val_image_dir}/patch_info.csv'
    val_video_info_csv = f'{config.val_image_dir}/video_info.csv'
    test_patch_info_csv = f'{config.test_image_dir}/patch_info.csv'
    test_video_info_csv = f'{config.test_image_dir}/video_info.csv'

    train_dataset = MultiJODDataset(
        data_dir=config.train_image_dir,
        patch_csv=train_patch_info_csv,
        video_csv=train_video_info_csv,
    )

    print(f'len(train_dataset) {len(train_dataset)}')

    val_dataset = MultiJODDataset(
        data_dir=config.val_image_dir,
        patch_csv=val_patch_info_csv,
        video_csv=val_video_info_csv,
    )
    
    test_dataset = MultiJODDataset(
        data_dir=config.test_image_dir,
        patch_csv=test_patch_info_csv,
        video_csv=test_video_info_csv,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
        persistent_workers=False
    )


    # CUDAPrefetcher : asynchronously loading the next batch to GPU
    train_prefetcher = CUDAPrefetcher(train_loader, config.device)
    val_prefetcher   = CUDAPrefetcher(val_loader, config.device)
    test_prefetcher  = CUDAPrefetcher(test_loader, config.device)

    return train_prefetcher, val_prefetcher, test_prefetcher


def train_one_epoch(model, train_prefetcher, optimizer, criterion, writer, epoch):
    """Run one full epoch of training."""
    model.train()
    train_prefetcher.reset()  # Reset the CUDA prefetcher so it starts from the first batch

    total_train_loss = 0.0
    batch_count = 0

    batch = train_prefetcher.next()  # Grab the first batch
    while batch is not None:
        # Extract data (already on GPU if your CUDAPrefetcher does .to(device))
        print(f'')
        image = batch[0]
        velocity = batch[1]
        bitrate = batch[2]
        jod = batch[3] #.to(config.device, non_blocking=True)
        # image, velocity, bitrate, jod_tensor are already on device

        # Forward/backward
        optimizer.zero_grad() # 1. zero out gradients, otherwise, gradients accumulate over epochs
        preds = model(image, velocity, bitrate)
        loss = criterion(preds, jod)
        loss.backward() # 2.
        optimizer.step() # 3. 1-3 important

        total_train_loss += loss.item()
        batch_count += 1

        batch = train_prefetcher.next()

    avg_train_loss = total_train_loss / batch_count
    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f}")

    if writer is not None:
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)


def validate_one_epoch(model, val_prefetcher, criterion, writer, epoch):
    """Run one full epoch of validation."""
    model.eval()
    val_prefetcher.reset()

    total_val_loss = 0.0
    batch_count = 0

    with torch.no_grad():
        batch = val_prefetcher.next()
        while batch is not None:
            image = batch[0]
            velocity = batch[1]
            bitrate = batch[2]
            jod = batch[3]

            preds = model(image, velocity, bitrate)
            loss = criterion(preds, jod)
            total_val_loss += loss.item()
            batch_count += 1

            batch = val_prefetcher.next()

    avg_val_loss = total_val_loss / batch_count
    print(f"[Epoch {epoch+1}] Val   Loss: {avg_val_loss:.4f}")

    if writer is not None:
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)

    return avg_val_loss



if __name__ == "__main__":
     # 1) Load dataset
    train_prefetcher, val_prefetcher, test_prefetcher = load_dataset()
    batches = len(train_prefetcher)
    print(f'Total number of batches {batches}')
    print("Load train dataset and valid dataset successfully.")

    # 2) Build model
    model = MultiJODModel().to(config.device)
    print("Build MultiJODModel model successfully.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr) # TODO: Adma?
    criterion = nn.MSELoss()

    # 3) TensorBoard summary writer
    results_dir = os.path.join("results", config.date_folder, config.hrmin_folder)
    os.makedirs(results_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(results_dir, "logs"))

    print(f"[Info] Starting training with patience={config.early_stopping_patience} ...")
    best_val_loss = float("inf")
    no_improve_count = 0
    best_model_path = None

    for epoch in range(config.start_epoch, config.epochs):
        train_one_epoch(model, train_prefetcher, optimizer, criterion, writer, epoch)
        val_loss = validate_one_epoch(model, val_prefetcher, criterion, writer, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve_count = 0
            best_model_path  = os.path.join(results_dir, f"model_best_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), best_model_path )
            print(f"Best model updated! Saved at {best_model_path }")
        else:
            no_improve_count += 1
            print(f"Validation loss did not improve for {no_improve_count} epoch(s).")
        
        if no_improve_count >= config.early_stopping_patience:
            print("Early stopping triggered!")
            break

    print("[Info] Training complete.")
    writer.close()

    if best_model_path is not None:
        # Load best model weights
        model.load_state_dict(torch.load(best_model_path))
        
        # Save again as the final model for clarity
        final_model_path = os.path.join(results_dir, "model_final.pth")
        torch.save(model.state_dict(), final_model_path)
        print(f"Saved best model as final model to {final_model_path}")

    # batch_time = AverageMeter("Time", ":6.3f")
    # data_time = AverageMeter("Data", ":6.3f")
    # losses = AverageMeter("Loss", ":6.6f")
    # psnres = AverageMeter("PSNR", ":4.2f")
    # progress = ProgressMeter(batches, [batch_time, data_time, losses, psnres], prefix=f"Epoch: [{epoch + 1}]")