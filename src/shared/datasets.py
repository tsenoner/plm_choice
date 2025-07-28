import h5py
import numpy as np
import polars as pl
from torch.utils.data import Dataset, DataLoader


class H5PyDataset(Dataset):
    def __init__(
        self,
        data: pl.DataFrame,
        file_path: str,
        param_name: str,
    ):
        self.param_name = param_name
        self.file_path = file_path
        self.file = None

        self.queries = data.select("query").to_series().to_numpy()
        self.targets = data.select("target").to_series().to_numpy()
        self.param_values = (
            data.select(param_name).to_series().to_numpy().astype(np.float32)
        )

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        if self.file is None:
            # Optimized HDF5 file opening with larger cache
            self.file = h5py.File(
                self.file_path,
                "r",
                swmr=True,
            )

        query_id = self.queries[idx]
        target_id = self.targets[idx]
        param_value = self.param_values[idx]

        # Get embeddings with optional caching
        query_emb_np = self._get_embedding(query_id)
        target_emb_np = self._get_embedding(target_id)

        return query_emb_np, target_emb_np, param_value

    def _get_embedding(self, protein_id: str) -> np.ndarray:
        """Get embedding"""
        # Load from HDF5
        embedding = self.file[protein_id][:].flatten().astype(np.float32)

        return embedding

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def get_embedding_size(hdf_file: str) -> int:
    """Reads the shape of the first dataset in the HDF5 file and returns its total size as an int."""
    with h5py.File(hdf_file, "r", rdcc_nbytes=32 * 1024 * 1024) as hdf:
        first_key = next(iter(hdf))
        embedding_shape = hdf[first_key].shape
        return int(np.prod(embedding_shape))


# --- Helper function for loading and filtering data --- #
def _load_and_filter_data(file_path, hdf_file, param_name):
    """Loads a parquet file, keeps necessary columns, removes NaNs, and filters based on HDF5 keys."""
    print(f"Loading and filtering data from: {file_path}")

    # Only support parquet files
    if not file_path.endswith(".parquet"):
        raise ValueError(f"Only parquet files are supported. Got: {file_path}")

    try:
        df = pl.read_parquet(file_path, columns=["query", "target", param_name])
    except Exception as e:
        raise ValueError(
            f"Error reading {file_path}. Ensure 'query', 'target', and '{param_name}' columns exist. Original error: {e}"
        )

    initial_rows = df.height
    df = df.drop_nulls(subset=[param_name])
    if df.height < initial_rows:
        print(
            f"Dropped {initial_rows - df.height} rows with null values in '{param_name}' column."
        )

    # Filter valid proteins based on keys present in the HDF5 file
    try:
        with h5py.File(hdf_file, "r") as hdf:
            valid_keys = set(hdf.keys())
    except Exception as e:
        raise IOError(
            f"Error opening or reading HDF5 file {hdf_file}. Original error: {e}"
        )

    filtered_df = df.filter(
        pl.col("query").is_in(valid_keys) & pl.col("target").is_in(valid_keys)
    )
    if filtered_df.height < df.height:
        print(
            f"Dropped {df.height - filtered_df.height} rows due to missing keys in HDF5 file {hdf_file}."
        )

    if filtered_df.height == 0:
        print(f"Warning: No valid data remaining after filtering for {file_path}.")

    return filtered_df


# ------------------------------------------------------ #


def create_single_loader(
    parquet_file: str,
    hdf_file: str,
    param_name: str,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 4,
) -> DataLoader:
    """Creates an optimized DataLoader for a single parquet dataset."""
    data = _load_and_filter_data(parquet_file, hdf_file, param_name)

    dataset = H5PyDataset(
        data,
        hdf_file,
        param_name,
    )

    persistent_workers = num_workers > 0

    # Calculate optimal prefetch factor
    prefetch_factor = max(2, min(6, batch_size // 1024 + 2)) if num_workers > 0 else 2

    if persistent_workers:
        print(
            f"Using {num_workers} persistent workers with prefetch_factor={prefetch_factor}"
        )
    else:
        print(f"Not using persistent workers (num_workers={num_workers}).")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        pin_memory=True,
        prefetch_factor=prefetch_factor,
    )
    print("Optimized DataLoader initialized with pin_memory=True.")
    return loader
