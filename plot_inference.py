import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns


def get_fps_res(inference_output_dir):
    predicted_res_path = f"{inference_output_dir}/{datatype}_predicted_res.py"
    target_res_path = f"{inference_output_dir}/{datatype}_target_res.py"
    predicted_fps_path = f"{inference_output_dir}/{datatype}_predicted_fps.py"
    target_fps_path = f"{inference_output_dir}/{datatype}_target_fps.py"
    
    data = {}

    for file_path, key in zip(
        [predicted_res_path, target_res_path, predicted_fps_path, target_fps_path],
        ["predicted_res", "target_res", "predicted_fps", "target_fps"]
    ):
        with open(file_path, "r") as f:
            exec(f.read(), data)

    pred_res = np.array(data["predicted_res"])
    target_res = np.array(data["target_res"])
    pred_fps = np.array(data["predicted_fps"])
    target_fps = np.array(data["target_fps"])
    
    # count_target_1080 = np.sum(target_res == 864)
    # count_predicted_1080 = np.sum(pred_res == 864)
    # print(f'count_target_864 {count_target_1080}, count_predicted_864 {count_predicted_1080}')  # Output: 3
    return pred_res, target_res, pred_fps, target_fps

def quiver_plot_fps_res(pred_res, target_res, pred_fps, target_fps, path='', SAVE=False, SHOW=False):
    # Compute quiver arrow components
    dx = target_res - pred_res
    dy = target_fps - pred_fps

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot quiver arrows where predicted != target
    mask = (dx != 0) | (dy != 0)
    ax.quiver(pred_res[mask], pred_fps[mask], dx[mask], dy[mask], angles='xy', scale_units='xy', scale=1, color='b', alpha=0.1,
              headwidth=8, headlength=10, headaxislength=8  # Make arrowhead bigger
              )

    # Plot points where predicted == target
    mask_identical = (dx == 0) & (dy == 0)
    ax.scatter(pred_res[mask_identical], pred_fps[mask_identical], color='r', label='Targets, predictions align')

    # Labels and formatting
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Framerate (FPS)")
    ax.set_title(f"Predicted vs Target FPS and Resolution (point to target)")
    ax.grid(True)
    ax.legend()
    
    ax.set_xticks(resolution_ticks)
    ax.set_yticks(fps_ticks)

    if SAVE:
        plt.savefig(f'{basepath}/quiver.png')
    if SHOW:
        print(f'show {SHOW}')
        plt.show()


def heatmap_fps_res(pred_res, target_res, pred_fps, target_fps, path='', SAVE=False, SHOW=False):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fps_confusion_matrix = np.zeros((len(fps_ticks), len(fps_ticks)), dtype=int)

    # --- 2. FPS Heatmap ---
    for i, true_fps in enumerate(fps_ticks):
        for j, pred_fps_value in enumerate(fps_ticks):
            fps_confusion_matrix[i, j] = np.sum((target_fps == true_fps) & (pred_fps == pred_fps_value))

    fps_confusion_matrix = np.flipud(fps_confusion_matrix.T)

    sns.heatmap(fps_confusion_matrix, annot=True, cmap="Greens", fmt="d", 
                xticklabels=fps_ticks, yticklabels=fps_ticks[::-1], ax=axes[0])

    axes[0].set_xlabel("Ground Truth Resolution")
    axes[0].set_ylabel("Predicted Resolution")
    axes[0].set_title(f"Resolution Prediction Heatmap\n{datatype}")

    # --- 2. Resolution Heatmap ---
    resolution_confusion_matrix = np.zeros((len(resolution_ticks), len(resolution_ticks)), dtype=int)
    for i, true_res in enumerate(resolution_ticks):
        for j, pred_res_value in enumerate(resolution_ticks):
            resolution_confusion_matrix[i, j] = np.sum((target_res == true_res) & (pred_res == pred_res_value))

    resolution_confusion_matrix = np.flipud(resolution_confusion_matrix.T)

    sns.heatmap(resolution_confusion_matrix, annot=True, cmap="Greens", fmt="d", 
                xticklabels=resolution_ticks, yticklabels=resolution_ticks[::-1], ax=axes[1])

    axes[1].set_xlabel("Ground Truth Resolution")
    axes[1].set_ylabel("Predicted Resolution")
    axes[1].set_title(f"Resolution Prediction Confusion Heatmap\n{datatype}")

    if SAVE:
        plt.savefig(f'{basepath}/heatmap.png')
    if SHOW:
        plt.show()


if __name__ == "__main__":
    SAVE = True # True False
    SHOW = True 
    datatype = 'dropjod' 
    basepath = f'output/train_model1'
    fps_ticks = list(range(30, 121, 10))
    # resolution_ticks = [360, 480, 720, 864, 1080]
    resolution_ticks = [360, 480, 720, 864, 1080]

    # Loop through each subfolder in the parent directory
    pred_res, target_res, pred_fps, target_fps = get_fps_res(basepath)
    # count_target_1080 = np.sum(target_res == 864)
    # count_predicted_1080 = np.sum(pred_res == 864)
    # print(f'target_res {target_res.shape}')
    # print(f'!!!count_target_864 {count_target_1080}, count_predicted_864 {count_predicted_1080}')  # Output: 3

    path = f'{basepath}'
    print(f'\ndatatype {datatype}, path {path}')
    quiver_plot_fps_res(pred_res, target_res, pred_fps, target_fps, path=path, SAVE=SAVE, SHOW=False)

    # Count how many times predicted resolution and FPS matches target resolution and FPS
    res_match_counts = {res: np.sum((target_res == res) & (pred_res == res)) for res in resolution_ticks}
    fps_match_counts = {fps: np.sum((target_fps == fps) & (pred_fps == fps)) for fps in fps_ticks}

    res_match_df = pd.DataFrame(list(res_match_counts.items()), columns=['Resolution', 'Matches'])
    fps_match_df = pd.DataFrame(list(fps_match_counts.items()), columns=['FPS', 'Matches'])
    # print(f'res_match_df \n{res_match_df}')
    # print(f'fps_match_df \n{fps_match_df}')

    heatmap_fps_res(pred_res, target_res, pred_fps, target_fps, path=path, SAVE=SAVE, SHOW=True)
