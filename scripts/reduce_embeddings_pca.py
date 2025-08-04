import argparse
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import warnings


def reduce_embeddings_with_pca(
    input_file: Path,
    output_file: Path,
    plot_file: Path,
    n_components: int,
    variance_summary_file: Path,
):
    """
    Reduces embeddings in an HDF5 file using PCA and saves the result.
    Also plots the cumulative explained variance.
    """
    logging.info(f"Processing {input_file.name}...")

    with h5py.File(input_file, "r") as f_in:
        protein_ids = list(f_in.keys())
        embeddings = np.array([f_in[pid][:] for pid in protein_ids])

        if embeddings.ndim == 3:
            embeddings = embeddings.squeeze(axis=1)

    # Use float64 for more precision during PCA
    embeddings = embeddings.astype(np.float64)

    # Handle non-finite values
    if not np.all(np.isfinite(embeddings)):
        logging.warning(
            f"Non-finite values (NaN or inf) found in {input_file.name}. Replacing with zeros."
        )
        embeddings = np.nan_to_num(embeddings)

    logging.info(
        f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}."
    )

    # Perform PCA
    pca = PCA(n_components=None)
    logging.info("Fitting PCA...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pca.fit(embeddings)
    logging.info("PCA fitting complete.")

    # Calculate cumulative variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Get variance at n_components
    variance_at_n_components = cumulative_variance[n_components - 1]
    logging.info(
        f"Cumulative variance with {n_components} components: {variance_at_n_components:.4f}"
    )

    # Write variance to summary file
    with open(variance_summary_file, "a") as f:
        f.write(f"{input_file.stem}: {variance_at_n_components:.4f}\n")

    # Plot cumulative explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_variance)
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title(f"Cumulative Explained Variance for {input_file.stem}")

    # Add cutoff line and annotation
    plt.axvline(
        x=n_components,
        color="r",
        linestyle="--",
        label=f"Cutoff at {n_components} components",
    )
    plt.axhline(
        y=variance_at_n_components,
        color="g",
        linestyle="--",
        xmax=n_components / len(cumulative_variance),
    )
    plt.annotate(
        f"{variance_at_n_components:.2%}",
        xy=(n_components, variance_at_n_components),
        xytext=(n_components + 5, variance_at_n_components - 0.05),
        arrowprops=dict(facecolor="black", shrink=0.05),
    )

    plt.legend()
    plt.grid(True)
    plt.savefig(plot_file)
    plt.close()
    logging.info(f"Saved explained variance plot to {plot_file}")

    logging.info("Transforming embeddings to reduced dimensions...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Center the data before projecting
        centered_embeddings = embeddings - pca.mean_
        reduced_embeddings = centered_embeddings @ pca.components_[:n_components].T
    logging.info("Transformation complete.")

    # Save reduced embeddings
    with h5py.File(output_file, "w") as f_out:
        for i, protein_id in enumerate(protein_ids):
            f_out.create_dataset(
                protein_id, data=reduced_embeddings[i].astype(np.float32)
            )
    logging.info(f"Saved reduced embeddings to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Reduce embedding dimensions using PCA."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default="data/processed/sprot_pre2024_subset/embeddings",
        help="Directory containing the original embedding HDF5 files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default="data/processed/sprot_pre2024_subset_pca",
        help="Directory to save the reduced embeddings and plots.",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=128,
        help="Number of principal components to keep.",
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "reduce_embeddings_pca.log"
    if log_file.exists():
        log_file.unlink()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    output_embeddings_dir = args.output_dir / "embeddings"
    output_plots_dir = args.output_dir / "plots"

    output_embeddings_dir.mkdir(parents=True, exist_ok=True)
    output_plots_dir.mkdir(parents=True, exist_ok=True)

    variance_summary_file = output_plots_dir / "variance_summary.txt"
    # Clear summary file if it exists
    if variance_summary_file.exists():
        variance_summary_file.unlink()

    logging.info(f"Input directory: {args.input_dir}")
    logging.info(f"Output embeddings directory: {output_embeddings_dir}")
    logging.info(f"Output plots directory: {output_plots_dir}")
    logging.info(f"Variance summary will be saved to: {variance_summary_file}")

    for input_file in sorted(list(args.input_dir.glob("*.h5"))):
        output_file = output_embeddings_dir / input_file.name
        plot_file = output_plots_dir / f"{input_file.stem}_explained_variance.png"

        reduce_embeddings_with_pca(
            input_file, output_file, plot_file, args.n_components, variance_summary_file
        )

    logging.info("All files processed.")


if __name__ == "__main__":
    main()
