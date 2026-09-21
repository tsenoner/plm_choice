import os
from pathlib import Path

import h5py
import numpy as np
import polars as pl
from torch.utils.data import DataLoader, Dataset

from shared.protein_cohort import cohort_size, load_excluded_proteins, restrict_to_cohort


class H5PyDataset(Dataset):
    """Pairs of protein embeddings and one target value.

    Reading is either lazy (one HDF5 lookup per protein per pair, the original behaviour)
    or preloaded once into a matrix. Preloading is what makes a large arm trainable on a
    shared filesystem: the loader touches each embedding once per pair, so a 5.6 GB arm on
    GPFS ran at 0.24 iterations/s against 23 on node-local disk -- a 12 h walltime that
    produced two epochs. Preloading turns that into one sequential read plus array indexing,
    and removes the dependency on a node happening to have spare local disk.

    It changes no number: the same float32 values are returned either way, which
    ``tests/test_datasets_preload.py`` pins on a fixture.

    Enable with ``preload=True`` or ``PLM_PRELOAD_EMBEDDINGS=1``. Cost is the arm's size in
    RAM (5.4 GB for the widest, esm2_3b at 2560 dimensions over 526,871 proteins), so it is
    opt-in rather than default.
    """

    def __init__(
        self,
        data: pl.DataFrame,
        file_path: str,
        param_name: str,
        preload: bool | None = None,
    ):
        self.param_name = param_name
        self.file_path = file_path
        self.file = None
        if preload is None:
            preload = os.environ.get("PLM_PRELOAD_EMBEDDINGS", "") == "1"
        self._store: dict[str, np.ndarray] | None = None

        self.queries = data.select("query").to_series().to_numpy()
        self.targets = data.select("target").to_series().to_numpy()
        self.param_values = (
            data.select(param_name).to_series().to_numpy().astype(np.float32)
        )
        if preload:
            needed = set(self.queries.tolist()) | set(self.targets.tolist())
            with h5py.File(file_path, "r") as hdf:
                self._store = {pid: hdf[pid][:].flatten().astype(np.float32) for pid in needed}
            print(f"Preloaded {len(self._store):,} embeddings from {file_path} into memory.")

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        if self._store is None and self.file is None:
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
        """Get one embedding, from memory when preloaded and from the file otherwise."""
        if self._store is not None:
            return self._store[protein_id]
        return self.file[protein_id][:].flatten().astype(np.float32)

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
        ) from e

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
        raise OSError(
            f"Error opening or reading HDF5 file {hdf_file}. Original error: {e}"
        ) from e

    # Restrict to the cohort shared by every embedding arm. Without this each arm
    # keeps whatever proteins its own HDF5 happens to contain, and since a pair is
    # dropped when EITHER protein is missing, the arms end up scored on different --
    # and differently sized -- test sets. See shared.protein_cohort for why this is a
    # load-time filter over a committed id list rather than a deletion, and why
    # completing the arms cannot substitute for it (ESM-1b's 1022-token cap).
    # No freeze committed => empty exclusion => this is a no-op.
    excluded = load_excluded_proteins()
    if excluded:
        valid_keys, summary = restrict_to_cohort(valid_keys, excluded)
        print(summary.describe(Path(hdf_file).stem))
        # Subtracting the exclusion is necessary but NOT sufficient. The filter only
        # removes ids the freeze names; it cannot notice an arm that is short for a
        # reason the freeze does not describe -- a stale or interrupted embedding run,
        # a torn .fai rewrite (see 447d875), or simply an --h5-dir pointing at arms
        # that were never cut to the cohort. Each leaves this arm on a different test
        # set than its peers, which is the exact defect the freeze exists to prevent,
        # and the count check is the only thing that can see it.
        cohort_size(warn_if_not=summary.kept, label=Path(hdf_file).stem)

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
    preload: bool | None = None,
) -> DataLoader:
    """Creates an optimized DataLoader for a single parquet dataset."""
    data = _load_and_filter_data(parquet_file, hdf_file, param_name)

    dataset = H5PyDataset(
        data,
        hdf_file,
        param_name,
        preload=preload,
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
