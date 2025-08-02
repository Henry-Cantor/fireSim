import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, input_seq, target_multi, region, horizons=None):

    model.eval()
    with torch.no_grad():
        # Add batch dimension
        inputs = input_seq.unsqueeze(0)  # shape: [1, seq_len, feature_dim]
        preds_tensor = model(inputs)     # shape: [1, num_horizons]

        preds = preds_tensor.cpu().numpy().flatten()
        gt = target_multi.cpu().numpy().flatten()
        input_plume = inputs.cpu().numpy().squeeze(0)  # [seq_len, feature_dim]

        metrics = {}
        mse_list = []
        mae_list = []

        for i in range(len(gt)):
            mse = mean_squared_error([gt[i]], [preds[i]])
            mae = mean_absolute_error([gt[i]], [preds[i]])
            mse_list.append(mse)
            mae_list.append(mae)
            horizon = horizons[i] if horizons is not None else i + 1
            metrics[f"mse_horizon_{horizon}d"] = mse
            metrics[f"mae_horizon_{horizon}d"] = mae

        metrics["mse_avg"] = np.mean(mse_list)
        metrics["mae_avg"] = np.mean(mae_list)

    print(f"[EVAL] Region {region} -- " +
          ", ".join([f"H{h}d MSE={metrics[f'mse_horizon_{h}d']:.4f}, MAE={metrics[f'mae_horizon_{h}d']:.4f}" for h in (horizons or range(1,len(gt)+1))]) +
          f", AVG MSE={metrics['mse_avg']:.4f}, AVG MAE={metrics['mae_avg']:.4f}")

    return input_plume, preds, gt, metrics
