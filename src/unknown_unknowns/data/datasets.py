import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


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
            # Ensure file is opened in worker process if using num_workers > 0
            self.file = h5py.File(self.file_path, "r", swmr=True)

        row = self.data.iloc[idx]
        # Return numpy arrays again - this was better for aten::copy_
        query_emb_np = self.file[row["query"]][:].flatten().astype(np.float32)
        target_emb_np = self.file[row["target"]][:].flatten().astype(np.float32)
        param_value_np = np.float32(row[self.param_name])

        return query_emb_np, target_emb_np, param_value_np

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def get_embedding_size(hdf_file: str) -> int:
    """Reads the shape of the first dataset in the HDF5 file and returns its total size as an int."""
    with h5py.File(hdf_file, "r") as hdf:
        first_key = next(iter(hdf))
        embedding_shape = hdf[first_key].shape
        return int(np.prod(embedding_shape))


# --- Helper function for loading and filtering data --- #
def _load_and_filter_data(file_path, hdf_file, param_name):
    """Loads a CSV, keeps necessary columns, removes NaNs, and filters based on HDF5 keys."""
    print(f"Loading and filtering data from: {file_path}")
    try:
        df = pd.read_csv(file_path, usecols=["query", "target", param_name])
    except ValueError as e:
        raise ValueError(
            f"Error reading {file_path}. Ensure 'query', 'target', and '{param_name}' columns exist. Original error: {e}"
        )

    initial_rows = len(df)
    df = df.dropna(subset=[param_name])
    if len(df) < initial_rows:
        print(
            f"Dropped {initial_rows - len(df)} rows with NaN in '{param_name}' column."
        )

    # Filter valid proteins based on keys present in the HDF5 file
    try:
        with h5py.File(hdf_file, "r") as hdf:
            valid_keys = set(hdf.keys())
    except Exception as e:
        raise IOError(
            f"Error opening or reading HDF5 file {hdf_file}. Original error: {e}"
        )

    filtered_df = df[
        df["query"].isin(valid_keys) & df["target"].isin(valid_keys)
    ].copy()
    if len(filtered_df) < len(df):
        print(
            f"Dropped {len(df) - len(filtered_df)} rows due to missing keys in HDF5 file {hdf_file}."
        )

    if len(filtered_df) == 0:
        print(f"Warning: No valid data remaining after filtering for {file_path}.")

    return filtered_df


# ------------------------------------------------------ #


def create_single_loader(
    csv_file: str,
    hdf_file: str,
    param_name: str,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Creates a DataLoader for a single CSV dataset."""
    data = _load_and_filter_data(csv_file, hdf_file, param_name)
    dataset = H5PyDataset(data, hdf_file, param_name)

    persistent_workers = num_workers > 0
    if persistent_workers:
        print(f"Using {num_workers} persistent workers for DataLoader.")
    else:
        print(f"Not using persistent workers (num_workers={num_workers}).")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,  # Explicitly set pin_memory=True
        # pin_memory_device="mps" # Usually not needed, PyTorch handles default device
    )
    print(f"DataLoader initialized with pin_memory=True.")  # Add log
    return loader
