import torch
import torch.nn as nn
import torch.nn.functional as F

def get_NN():
    """
    Same as your previous CNN backbone, returning a 32-dim feature vector.
    We'll drop the final Sigmoid so that the feature is unbounded.
    """
    nnSequential = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2,2),

        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),  # final feature dim
        # remove nn.Sigmoid() so we get a wide range of feature values
    )
    return nnSequential

class MultiJODModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = get_NN()  # Returns [B, 32] feature
        # We'll combine [32 + 2] => 50 outputs
        # self.fc = nn.Sequential(
        #     nn.Linear(32 + 2, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 50)  # 50-dim JOD output
        # )
        self.fc = nn.Sequential(
            nn.Linear(32 + 2, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),  # Helps regularization on bigger data
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, 50)   # final output: 50-dim
        )

    def forward(self, image, velocity, bitrate):
        """
        image:  [B, 3, H, W]
        velocity: [B, 1]
        bitrate:  [B, 1]
        Returns [B, 50]
        """
        feats = self.cnn(image)  # [B, 32]
        # Concat velocity, bitrate
        extra = torch.cat([velocity, bitrate], dim=1)  # [B, 2]
        combined = torch.cat([feats, extra], dim=1)    # [B, 34]
        out = self.fc(combined)                        # [B, 50]
        return out
