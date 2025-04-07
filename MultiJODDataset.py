import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


class MultiJODDataset(Dataset):
    def __init__(self,
                 data_dir, 
                 patch_csv, 
                 video_csv,
                 patch_size=((64, 64)),
                 bitrates=(500, 1000, 1500, 2000)):
        """
        patch_csv: path to patch_info.csv
          columns: [patch_path, video_id, velocity, ...]
        video_csv: path to video_info.csv
          columns: [video_id, bitrate, jod_0, ..., jod_49]
        patches_dir: folder where patch images are located
        transform: optional torchvision transform for the patches
        bitrates: the set of bitrates you want to train on
        """
        self.data_dir = data_dir
        self.patch_df = pd.read_csv(patch_csv)
        self.video_df = pd.read_csv(video_csv)

        self.max_velocity = 0.31414 # 1.79153
        self.min_velocity = 7e-05
        self.min_bitrate = 500
        self.max_bitrate = 2000

        self.transform = transforms.Compose([
            transforms.Resize(patch_size),  # Resize images to 64x64
            transforms.ToTensor(),  # Convert images to PyTorch tensors
        ]) 

        # Build a dict: (video_id, bitrate) -> [50 JOD floats]
        self.video_dict = {}
        for _, row in self.video_df.iterrows():
            vid = row["video_id"]
            b   = row["bitrate"]
            jods = [row[f"jod_{i}"] for i in range(50)]  # Collect jod_0..jod_49
            self.video_dict[(vid, b)] = jods

        # Build a list of all (patch_idx, bitrate) pairs
        self.pairs = []
        for i, patch_row in self.patch_df.iterrows():
            # print(f'i, patch_row {i, patch_row}')
            video_id = patch_row["video_id"] # like bedroom_path1_seg1_1
            # We only add a pair if (video_id, b) is in self.video_dict
            # That means we have JOD data for that combo
            for b in bitrates:
                if (video_id, b) in self.video_dict:
                    self.pairs.append((i, b)) # (row_1, bitrate)
        print(f'Total number of data {len(self.pairs)}, e.g. row_index, bitrate: {self.pairs[-1]}')
            # count += 1
            # if count >=3:
            #     break

    def __len__(self):
        return len(self.pairs)
    
    def normalize(self, sample, min_vals, max_vals):
        # print(f'val, min_vals, max_vals {sample, min_vals, max_vals}')
        sample = (sample - min_vals) / (max_vals - min_vals)
        sample = np.clip(sample, 0, 1)
        return round(sample, 3)


    def __getitem__(self, idx):
        """
        Returns:
          image_tensor: [C, H, W]
          velocity: scalar float
          bitrate: scalar float
          jod_array: [50] float (the ground truth)
        """
        patch_idx, bitrate = self.pairs[idx]

        patch_row = self.patch_df.iloc[patch_idx]
        patch_path = patch_row["patch_path"]     # e.g. "bedroom_path1_seg1_1_frame0.jpg"
        video_id   = patch_row["video_id"]       # bedroom_path1_seg1_1
        velocity   = patch_row["velocity"]
        patch_path_full = os.path.join(self.data_dir, 'patches', patch_path)
        # print(f'patch_path_full {patch_path_full}')
        
        image = Image.open(patch_path_full).convert("RGB")

        if self.transform:
            image = self.transform(image)  # e.g. transforms.ToTensor()

        jod_array = self.video_dict[(video_id, bitrate)]  # shape [50]
        jod_tensor = torch.tensor(jod_array, dtype=torch.float)   # [50]
        
        # normalize
        bitrate = self.normalize(bitrate, self.min_bitrate, self.max_bitrate)
        velocity = self.normalize(velocity, self.min_velocity, self.max_velocity)

        velocity   = torch.tensor([velocity], dtype=torch.float)  # shape [1]
        bitrate    = torch.tensor([bitrate],  dtype=torch.float)  # shape [1]

        return image, velocity, bitrate, jod_tensor