import os
import random
import numpy as np
import torch
from torch.backends import cudnn
from datetime import datetime


# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True

mode = "debug" # "train"

if mode == "train" or mode == "debug":
    if mode == "train":
        ML_regression = r'C:\Users\15142\Projects\VRR\Data\VRRML\ML_regression'
        batch_size = 128
        epochs = 200
    if mode == "debug":
        ML_regression = r'C:\Users\15142\Projects\VRR\Data\VRRML\ML_regression\debug' # 100 train data, 50 test data
        batch_size = 30
        epochs = 5
    
    train_image_dir = f"{ML_regression}/train"
    val_image_dir = f"{ML_regression}/val"
    test_image_dir = f"{ML_regression}/test"

    test_batch_size = batch_size * 2
    num_workers = 4

    # Incremental training and migration training
    start_epoch = 0
    resume = ""
    lr = 3e-4 # 0.0003 
    early_stopping_patience = 10

    now = datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    hrmin_folder = now.strftime("%H_%M")