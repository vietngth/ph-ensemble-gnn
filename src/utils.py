import numpy as np
import os 
import csv

def normalize(inputs):
    return np.array([eq / np.maximum(np.max(np.abs(eq)), 1e-8) for eq in inputs])

def compute_mape(preds, targets, epsilon=1e-4):
    # Mask near-zero labels to avoid division by small numbers
    masked_labels = np.where(np.abs(targets) < epsilon, np.nan, targets)

    # Compute absolute percentage error
    mape = np.abs((preds - targets) / masked_labels)

    # Compute the mean over the specified axes, ignoring NaNs
    return np.nanmean(mape, axis=(0, 2, 3)) * 100

def log_metrics(**kwargs):
    window = kwargs["window"]
    log_path = kwargs["log_path"]
    exp = kwargs["exp"]
    
    mae = kwargs["mae"]
    mse = kwargs["mse"]
    rmse = kwargs["rmse"]
    mape = kwargs["mape"]

    mae = [np.round(val, 2) for val in mae]
    mse = [np.round(val, 2) for val in mse]
    rmse = [np.round(val, 2) for val in rmse]
    mape = [np.round(val, 2) for val in mape]

    metrics_log_path = os.path.join(log_path, "metrics_log")
    os.makedirs(metrics_log_path, exist_ok=True)

    print("path to write:", metrics_log_path)
    if exp == "forecasting":
        mae_path = os.path.join(metrics_log_path, f'mae_{kwargs["horizon"]}.csv')
        mse_path = os.path.join(metrics_log_path, f'mse_{kwargs["horizon"]}.csv')
        rmse_path = os.path.join(metrics_log_path, f'rmse_{kwargs["horizon"]}.csv')
        mape_path = os.path.join(metrics_log_path, f'mape_{kwargs["horizon"]}.csv')
        write_individual_metric_tf(mae_path, "MAE", mae, **kwargs)
        write_individual_metric_tf(mse_path, "MSE", mse, **kwargs)
        write_individual_metric_tf(rmse_path, "RMSE", rmse, **kwargs)
        write_individual_metric_tf(mape_path, "MAPE", mape, **kwargs)
    else:
        seed = kwargs["data_seed"]
        mae_path = os.path.join(metrics_log_path, f"mae_{window}_{seed}.csv")
        mse_path = os.path.join(metrics_log_path, f"mse_{window}_{seed}.csv")
        rmse_path = os.path.join(metrics_log_path, f"rmse_{window}_{seed}.csv")
        write_individual_metric(mae_path, "MAE", mae, **kwargs)
        write_individual_metric(mse_path, "MSE", mse, **kwargs)
        write_individual_metric(rmse_path, "RMSE", rmse, **kwargs)
    print("Write succesfully.")
    return 1


def write_individual_metric(csv_path, metric_name, metric_val, **kwargs):
    headlines = ["Dataset", "Graph Order", "Top-K Experts", "Num Graphs", "Model Type", "Window Size", "Seed", "Geo_Features", "PGA", "PGV", "PSA03", "PSA1", "PSA3", f"Average {metric_name}"]
    with open(csv_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        if(os.stat(csv_path).st_size == 0):
            writer.writerow(headlines)
        writer.writerow([
            kwargs["dataset"],
            kwargs["graph_order"],
            kwargs["top_k"],
            kwargs["num_graphs"] if kwargs["dataset"] != "baseline" else 1,
            kwargs["aggregator"],
            str(int(kwargs["window"]//100)),
            kwargs["data_seed"],
            kwargs["geo_feat"],
            metric_val[0], metric_val[1], metric_val[2], metric_val[3], metric_val[4], np.round(np.mean(metric_val), 2),
        ]) 
    return csv_path

def write_individual_metric_tf(csv_path, metric_name, metric_val, **kwargs):
    headlines = ["Dataset", "Graph Order", "Top-K Experts", "Num Graphs", "Window_Size", "Horizon", "1", "2", "3", "4", "5", "6",
                 "7", "8", "9", "10", "11", "12", f"Average {metric_name}"]
    with open(csv_path, "a") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        if(os.stat(csv_path).st_size == 0):
            writer.writerow(headlines)
        writer.writerow([
            kwargs["dataset"],
            kwargs["graph_order"],
            kwargs["top_k"],
            kwargs["num_graphs"],
            kwargs["window"],
            kwargs["horizon"],
            metric_val[0], metric_val[1], metric_val[2], metric_val[3], metric_val[4],
            metric_val[5], metric_val[6], metric_val[7], metric_val[8], metric_val[9], 
            metric_val[10], metric_val[11],
            np.round(np.mean(metric_val), 2),
        ]) 
    return csv_path
