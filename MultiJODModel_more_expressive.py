import torch
import torch.nn as nn
import torch.nn.functional as F

def get_NN():
    """
    Expanded CNN backbone:
      - First conv block: (3->32, 32->64) instead of (3->16,16->32)
      - Second conv block: (64->64, 64->64) remains the same
      - Third conv block: (64->128, 128->128) remains the same
    Final embedding is still a 32-dim feature vector.
    """
    nnSequential = nn.Sequential(
        # --- Block 1 ---
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),   # Output: 1/2 spatial

        # --- Block 2 ---
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),   # Output: 1/4 spatial

        # --- Block 3 ---
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),   # Output: 1/8 spatial

        # Global average + flatten -> 128-dim
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),

        # Final projection to 32-dim
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        # no sigmoid -> unbounded
    )
    return nnSequential


class MultiJODModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = get_NN()  # Returns [B, 32] feature

        # We'll combine [32 + 2] => 34 inputs
        # MLP: now has one extra hidden layer (128 -> 128 -> 64 -> 50)
        self.fc = nn.Sequential(
            nn.Linear(34, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(128, 128),  # extra layer
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(64, 50)   # final output: 50-dim
        )

    def forward(self, image, velocity, bitrate):
        """
        image:    [B, 3, H, W]
        velocity: [B, 1]
        bitrate:  [B, 1]
        Returns [B, 50] (JOD predictions)
        """
        # 1) CNN feature extraction
        feats = self.cnn(image)  # [B, 32]

        # 2) Concat velocity & bitrate
        extra = torch.cat([velocity, bitrate], dim=1)  # [B, 2]
        combined = torch.cat([feats, extra], dim=1)    # [B, 34]

        # 3) Predict 50 JOD scores
        out = self.fc(combined)  # [B, 50]
        return out
