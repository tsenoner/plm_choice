import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class H5PyDataset(Dataset):
    def __init__(self, data, file_path, param_name):
        self.data = data
        self.param_name = param_name
        self.file_path = file_path
        self.file = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.file_path, "r", swmr=True)

        row = self.data.iloc[idx]
        query_emb = torch.tensor(
            self.file[row["query"]][:].flatten(), dtype=torch.float32
        )
        target_emb = torch.tensor(
            self.file[row["target"]][:].flatten(), dtype=torch.float32
        )
        param_value = torch.tensor(row[self.param_name], dtype=torch.float32)

        return query_emb, target_emb, param_value

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device:", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon) device")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def get_embedding_size(hdf_file):
    with h5py.File(hdf_file, "r") as hdf:
        first_key = next(iter(hdf))
        embedding_shape = hdf[first_key].shape
        return np.prod(embedding_shape)


def create_data_loaders(
    train_file, val_file, test_file, hdf_file, param_name, batch_size=128
):
    def load_and_filter_data(file_path):
        df = pd.read_csv(file_path, usecols=["query", "target", param_name])
        df = df.dropna(subset=[param_name])

        # Filter valid proteins
        with h5py.File(hdf_file, "r") as hdf:
            return df[df["query"].isin(hdf.keys()) & df["target"].isin(hdf.keys())]

    # Load datasets
    train_data = load_and_filter_data(train_file)
    val_data = load_and_filter_data(val_file)
    test_data = load_and_filter_data(test_file)

    # Create data loaders
    def create_loader(data, shuffle):
        dataset = H5PyDataset(data, hdf_file, param_name)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=4,
            persistent_workers=True,
        )

    return (
        create_loader(train_data, shuffle=True),
        create_loader(val_data, shuffle=False),
        create_loader(test_data, shuffle=False),
    )


def plot_scatter(predictions, targets, output_file):
    plt.figure(figsize=(10, 10))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], "r--")

    # Calculate metrics
    r2 = r2_score(targets, predictions)
    pearson_corr, _ = pearsonr(predictions, targets)

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True vs Predicted\nPearson: {pearson_corr:.4f}, R²: {r2:.4f}")
    plt.savefig(output_file)
    plt.close()

    return {
        "MSE": mean_squared_error(targets, predictions),
        "RMSE": np.sqrt(mean_squared_error(targets, predictions)),
        "MAE": mean_absolute_error(targets, predictions),
        "R2": r2,
        "Pearson": pearson_corr,
        "Spearman": spearmanr(predictions, targets)[0],
    }
