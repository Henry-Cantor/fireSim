# pm25_plume_forecast/visualization/map_plotter.py

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_maps(input_plume, pred_plume, gt_plume, region, date, output_dir="results/plots"):
    """
    Plot side-by-side maps of input plume, predicted plume, and ground truth plume.
    Also plot danger maps (e.g., PM2.5 > threshold).
    Saves plots to output_dir.
    """

    os.makedirs(output_dir, exist_ok=True)

    threshold = 35.0  # Example PM2.5 danger threshold (µg/m³)

    vmax = max(np.max(input_plume), np.max(pred_plume), np.max(gt_plume))
    vmin = min(np.min(input_plume), np.min(pred_plume), np.min(gt_plume))

    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: Continuous PM2.5 maps
    im0 = axs[0,0].imshow(input_plume, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0,0].set_title("Input Plume (Past PM2.5)")
    plt.colorbar(im0, ax=axs[0,0], fraction=0.046, pad=0.04)

    im1 = axs[0,1].imshow(pred_plume, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0,1].set_title("Predicted Plume")
    plt.colorbar(im1, ax=axs[0,1], fraction=0.046, pad=0.04)

    im2 = axs[0,2].imshow(gt_plume, cmap="viridis", vmin=vmin, vmax=vmax)
    axs[0,2].set_title("Ground Truth Plume")
    plt.colorbar(im2, ax=axs[0,2], fraction=0.046, pad=0.04)

    # Row 2: Danger maps (binary mask where PM2.5 > threshold)
    input_danger = (input_plume > threshold).astype(float)
    pred_danger = (pred_plume > threshold).astype(float)
    gt_danger = (gt_plume > threshold).astype(float)

    im3 = axs[1,0].imshow(input_danger, cmap="Reds", vmin=0, vmax=1)
    axs[1,0].set_title(f"Input Danger Map (> {threshold})")

    im4 = axs[1,1].imshow(pred_danger, cmap="Reds", vmin=0, vmax=1)
    axs[1,1].set_title(f"Predicted Danger Map (> {threshold})")

    im5 = axs[1,2].imshow(gt_danger, cmap="Reds", vmin=0, vmax=1)
    axs[1,2].set_title(f"Ground Truth Danger Map (> {threshold})")

    for ax in axs.flat:
        ax.axis("off")

    plt.suptitle(f"PM2.5 Plume Forecast - Region: {region} - Date: {date}", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(output_dir, f"{region}_{date}_plume_maps.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT] Saved plume maps to {out_path}")


def plot_performance(metrics_list, region, output_dir="results/plots"):
    """
    Plot model performance metrics (MSE and MAE) over time.
    metrics_list is a list of dicts with keys: 'date', 'mse', 'mae'
    """

    os.makedirs(output_dir, exist_ok=True)

    dates = [m["date"] for m in metrics_list]
    mses = [m["mse"] for m in metrics_list]
    maes = [m["mae"] for m in metrics_list]

    fig, ax1 = plt.subplots(figsize=(12,6))

    ax1.plot(dates, mses, label="MSE", color="tab:blue", marker="o")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("MSE", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    plt.xticks(rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(dates, maes, label="MAE", color="tab:orange", marker="x")
    ax2.set_ylabel("MAE", color="tab:orange")
    ax2.tick_params(axis='y', labelcolor="tab:orange")

    plt.title(f"Model Performance over Time - {region}")
    fig.tight_layout()

    out_path = os.path.join(output_dir, f"{region}_performance_over_time.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[PLOT] Saved performance plot to {out_path}")
