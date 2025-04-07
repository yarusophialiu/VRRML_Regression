from torch.utils.data import DataLoader
import torchvision.transforms as T
from MultiJODDataset import *

transform = T.Compose([
    T.Resize((128,128)),
    T.ToTensor()
])

# data_prefix_path = r'C:\Users\15142\Projects\VRR\Data\VRRML\ML_regression\2025-03-31\1441'
# dataset = MultiJODDataset(
#     data_prefix_path,
#     patch_csv=f"{data_prefix_path}/patch_info.csv",
#     video_csv=f"{data_prefix_path}/video_info.csv",
#     bitrates=[500, 1000, 1500, 2000],
# )
# loader = DataLoader(dataset, batch_size=16, shuffle=True)


path = 'room_path5_seg2_2_frame93. jpg'
path = path.split(".")[0]
print(path)