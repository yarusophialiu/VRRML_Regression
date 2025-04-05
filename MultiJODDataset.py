import os
import pandas as pd
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
                    self.pairs.append((i, b))
        print(f'self.pairs {len(self.pairs)} {self.pairs[-1]}')
            # count += 1
            # if count >=3:
            #     break

    def __len__(self):
        return len(self.pairs)

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
        patch_path = patch_row["patch_path"]     # e.g. "bedroom_path1_seg1_1_fr0.jpg"
        video_id   = patch_row["video_id"]
        velocity   = patch_row["velocity"]

        patch_path_full = os.path.join(self.data_dir, patch_path)
        image = Image.open(patch_path_full).convert("RGB")

        if self.transform:
            image = self.transform(image)  # e.g. transforms.ToTensor()

        jod_array = self.video_dict[(video_id, bitrate)]  # shape [50]

        # Convert everything to torch
        jod_tensor = torch.tensor(jod_array, dtype=torch.float)   # [50]
        velocity   = torch.tensor([velocity], dtype=torch.float)  # shape [1]
        bitrate    = torch.tensor([bitrate],  dtype=torch.float)  # shape [1]

        return image, velocity, bitrate, jod_tensor
    #     self.patch_df = pd.read_csv(patch_csv)
    #     video_df = pd.read_csv(video_csv)

    #     # Build a lookup: (video_id, bitrate) -> [jod_0..jod_49]
    #     self.video_dict = {}
    #     for _, row in video_df.iterrows():
    #         vid = row["video_id"]
    #         b   = row["bitrate"]
    #         jods = [row[f"jod_{i}"] for i in range(50)]
    #         self.video_dict[(vid, b)] = jods
        
    #     self.transform = transforms.Compose([
    #         transforms.Resize(patch_size),  # Resize images to 64x64
    #         transforms.ToTensor(),  # Convert images to PyTorch tensors
    #     ])    

    #     self.patches_dir = patches_dir
    #     self.bitrates = bitrates

    #     # Build a list of all samples = (patch_idx, bitrate)
    #     # We'll produce one sample for each patch & each bitrate that exists
    #     self.samples = []
    #     for i, patch_row in self.patch_df.iterrows():
    #         print(f'i, patch_row {i, patch_row}')
    #         video_id = patch_row["video_id"]
    #         for b in self.bitrates:
    #             if (video_id, b) in self.video_dict:
    #                 self.samples.append((i, b))
    #         print(f'self.samples {self.samples}')
    #         break

    # def __len__(self):
    #     return len(self.samples)

    # def __getitem__(self, idx):
    #     patch_idx, bitrate = self.samples[idx]
    #     row = self.patch_df.iloc[patch_idx]

    #     patch_path = row["patch_path"]
    #     video_id   = row["video_id"]
    #     velocity   = float(row["velocity"])

    #     # Load image
    #     # If patch_path is just a filename, prepend self.patches_dir
    #     img_path = os.path.join(self.patches_dir, patch_path)
    #     image = Image.open(img_path).convert("RGB")

    #     if self.transform:
    #         image = self.transform(image)

       
