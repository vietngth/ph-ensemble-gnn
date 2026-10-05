"""Regression metrics per output (the five intensity measures)."""
import numpy as np


def regression_metrics(pred, target):
    """Return per-output MAE, MSE and RMSE for arrays shaped [..., n_outputs]."""
    err = np.asarray(pred, np.float64) - np.asarray(target, np.float64)
    err = err.reshape(-1, err.shape[-1])
    mse = (err ** 2).mean(0)
    return dict(mae=np.abs(err).mean(0).tolist(), mse=mse.tolist(), rmse=np.sqrt(mse).tolist())
