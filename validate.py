import os 
import time
import torch

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
# from MultiJODModel_more_expressive import *
from dataset import CUDAPrefetcher
from train import load_dataset

import numpy as np
import pandas as pd

def get_fps_resolution_with_highest_jod(jod_scores, fps_values, resolution_values):
    """Return the (fps, resolution) that gives the highest JOD score."""
    # max_jod_idx = np.argmax(jod_scores)
    argmax_indices = np.argmax(jod_scores, axis=1)
    print(f'argmax_indices \n{argmax_indices}')
    # TODO: verify fps_indices and res_indices are correct
    fps_indices = argmax_indices // len(resolution_values)
    res_indices = argmax_indices % len(resolution_values)
    fps_array = np.array(fps_values)[fps_indices]
    res_array = np.array(resolution_values)[res_indices]
    # rows = [0, 1, 2, ..., N-1]
    # cols = [idx0, idx1, idx2, ..., idxN-1] ← from argmax_indices
    jod_array = jod_scores[np.arange(len(jod_scores)), argmax_indices]

    return fps_array, res_array, jod_array

# def compare_fps_resolution_with_ground_truth(pred_jod_scores, ground_truth_jod_scores, fps_values, resolution_values):
#     """Compare predicted fps x resolution with ground truth."""
#     # 1. Get predicted highest fps x resolution
#     predicted_fps, predicted_res, predicted_max_jod = get_fps_resolution_with_highest_jod(pred_jod_scores, fps_values, resolution_values)
#     # 2. Get ground truth highest fps x resolution
#     gt_fps, gt_res, gt_max_jod = get_fps_resolution_with_highest_jod(ground_truth_jod_scores, fps_values, resolution_values)
    
#     print(f"Predicted: fps={predicted_fps}, resolution={predicted_res}, JOD={predicted_max_jod}\n\n")
#     print(f"Ground Truth: fps={gt_fps}, resolution={gt_res}, JOD={gt_max_jod}")
        
#     # 3. Get combinations within 0.25 of the max JOD
#     valid_combinations = []
#     for i in range(len(fps_values)):
#         for j in range(len(resolution_values)):
#             print(f'')
#             if abs(pred_jod_scores[i * len(resolution_values) + j] - predicted_max_jod) <= 0.25:
#                 valid_combinations.append((fps_values[i], resolution_values[j], pred_jod_scores[i * len(resolution_values) + j]))
    
#     print(f"Combinations within 0.25 of predicted max JOD:")
#     for fps, res, jod in valid_combinations:
#         print(f"fps={fps}, resolution={res}, JOD={jod:.4f}")

    
def find_drop_jod(max_jods, jod_scores, fps_values, res_values, THRESHOLD=0.25):
    fps_grid = np.repeat(fps_values, len(res_values))     # shape (50,) e.g. 30,30..., 40,40,...
    res_grid = np.tile(res_values, len(fps_values))        # shape (50,) e.g. 360,480,...,1080,360,480,...,1080
    fps_x_res = fps_grid * res_grid                        # shape (50,) array of fps x resolution

    best_combos = []
    dropjod_fps = []
    dropjod_res = []
    # dropjod_jod = []

    for i in range(len(jod_scores)):
        jod_row = jod_scores[i]
        max_jod = max_jods[i]

        # Find indices where JOD is within 0.25 of the max
        candidate_indices = np.where(jod_row >= (max_jod - THRESHOLD))[0]

        # Among those, find the one with smallest fps × resolution
        candidate_fps_x_res = fps_x_res[candidate_indices] # valid jod's fps x resolution values
        best_idx = candidate_indices[np.argmin(candidate_fps_x_res)]

        best_fps = fps_grid[best_idx]
        best_res = res_grid[best_idx]
        best_jod = jod_row[best_idx]

        best_combos.append((best_fps, best_res, best_jod))
        dropjod_fps.append(best_fps)
        dropjod_res.append(best_res)

    # for i, (fps, res, jod) in enumerate(best_combos):
    #     print(f"Sample {i:03d}: Best fps={fps}, res={res}, jod={jod:.4f}")
    return best_combos, dropjod_fps, dropjod_res


def get_test_preds_targets(test_prefetcher, model):
    all_predicted = []
    all_ground_truth = []

    model.eval()
    test_prefetcher.reset()

    with torch.no_grad():
        batch = test_prefetcher.next()
        while batch is not None:
            image = batch[0]         # [B, 3, H, W]
            velocity = batch[1]      # [B, 1]
            bitrate = batch[2]       # [B, 1]
            gt_jod = batch[3]        # [B, 50]

            pred_jod = model(image, velocity, bitrate)  # [B, 50]

            # Move to CPU and convert to numpy
            all_predicted.append(pred_jod.cpu().numpy())
            all_ground_truth.append(gt_jod.cpu().numpy())

            batch = test_prefetcher.next()

    # Stack all batches into full arrays
    predicted_jod_scores = np.vstack(all_predicted)       # shape: [num of data, 50]
    ground_truth_jod_scores = np.vstack(all_ground_truth) # shape: [num of data, 50]

    return predicted_jod_scores, ground_truth_jod_scores


def load_test_model(model_path):
    model = MultiJODModel().to(config.device)
    model.load_state_dict(torch.load(model_path))
    return model

def write_arrays_to_python_file(name, value, output_file, mode='w'):
    with open(output_file, mode) as f:
        f.write(f"{name} = ")
        # f.write(repr(value.tolist()))
        if isinstance(value, (np.ndarray, torch.Tensor)):
            value = value.tolist()
        f.write(repr(value))


if __name__ == "__main__":
    fps_values = [i for i in range(30, 121, 10)]  # Example fps range
    resolution_values = [360, 480, 720, 864, 1080]  # Example resolution values

    _, _, test_prefetcher = load_dataset(train=False, test=True)

    # Example JOD scores for a specific batch (5 fps * 5 resolution = 25 combinations)
    # predicted_jod_scores = np.random.uniform(1, 5, 25)  # Random predicted JOD scores
    # ground_truth_jod_scores = np.random.uniform(1, 5, 25)  # Random ground truth JOD scores
    model_name = 'model1'
    if model_name == 'model1':
        model_path = "results/2025-04-07/model1/model_best_epoch.pth"
    else:
        model_path = "results/2025-04-11/model2/model_best_19.pth"
    output_path = f'output/{config.mode}_{model_name}'
    os.makedirs(output_path, exist_ok=True)
    
    model = load_test_model(model_path)
    predicted_jod_scores, ground_truth_jod_scores = get_test_preds_targets(test_prefetcher, model)
    print(f'predicted_jod_scores {predicted_jod_scores}\n')
    print(f'ground_truth_jod_scores {ground_truth_jod_scores}\n')
    # write_arrays_to_python_file('predicted_jod_scores', predicted_jod_scores, f'{output_path}/maxjod_pred_jod.py')
    # write_arrays_to_python_file('target_jod_scores', ground_truth_jod_scores, f'{output_path}/maxjod_target_jod.py')

    # Get predicted and target highest fps x resolution
    predicted_fps, predicted_res, predicted_max_jod = get_fps_resolution_with_highest_jod(predicted_jod_scores, fps_values, resolution_values)
    gt_fps, gt_res, gt_max_jod = get_fps_resolution_with_highest_jod(ground_truth_jod_scores, fps_values, resolution_values)
    print(f'\npredicted_max_jod {predicted_max_jod}')
    print(f'\ngt_max_jod {gt_max_jod}')

    # fps, res, jod
    best_pred_combos, dropjod_pred_fps, dropjod_pred_res = find_drop_jod(predicted_max_jod, predicted_jod_scores, fps_values, resolution_values, THRESHOLD=0.25)
    best_gt_combos, dropjod_gt_fps, dropjod_gt_res = find_drop_jod(gt_max_jod, ground_truth_jod_scores, fps_values, resolution_values, THRESHOLD=0.25)
    print(f'best_combos {best_pred_combos[:5]}')

    write_arrays_to_python_file('predicted_fps', dropjod_pred_fps, f'{output_path}/dropjod_predicted_fps.py')
    write_arrays_to_python_file('predicted_res', dropjod_pred_res, f'{output_path}/dropjod_predicted_res.py')
    write_arrays_to_python_file('target_fps', dropjod_gt_fps, f'{output_path}/dropjod_target_fps.py')
    write_arrays_to_python_file('target_res', dropjod_gt_res, f'{output_path}/dropjod_target_res.py')

    # write_arrays_to_python_file('predicted_fps', predicted_fps, f'{output_path}/maxjod_predicted_fps.py')
    # write_arrays_to_python_file('predicted_res', predicted_res, f'{output_path}/maxjod_predicted_res.py')
    # write_arrays_to_python_file('target_fps', gt_fps, f'{output_path}/maxjod_target_fps.py')
    # write_arrays_to_python_file('target_res', gt_res, f'{output_path}/maxjod_target_res.py')



