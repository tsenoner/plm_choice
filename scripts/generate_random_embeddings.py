import argparse
import h5py
import numpy as np
import os


def generate_random_embeddings(template_h5_path, output_dir, dimensions):
    """
    Generates HDF5 files containing random embeddings for protein IDs found
    in a template HDF5 file.

    Args:
        template_h5_path (str): Path to the template HDF5 embedding file.
        output_dir (str): Directory to save the generated random embedding files.
        dimensions (list[int]): A list of embedding dimensions to generate.
    """
    print(f"Using template file: {template_h5_path}")
    print(f"Output directory: {output_dir}")
    print(f"Generating dimensions: {dimensions}")

    os.makedirs(output_dir, exist_ok=True)

    try:
        with h5py.File(template_h5_path, "r") as template_f:
            protein_ids = list(template_f.keys())
            print(f"Found {len(protein_ids)} protein IDs in the template.")

            for dim in dimensions:
                output_filename = f"random_{dim}.h5"
                output_path = os.path.join(output_dir, output_filename)
                print(f"Generating embeddings for dimension {dim} -> {output_path}...")

                with h5py.File(output_path, "w") as output_f:
                    for protein_id in protein_ids:
                        # Generate random embedding with standard normal distribution
                        random_embedding = np.random.randn(dim).astype(np.float16)
                        output_f.create_dataset(protein_id, data=random_embedding)
                print(f"Finished generating {output_filename}")

    except FileNotFoundError:
        print(f"Error: Template file not found at {template_h5_path}")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print("Random embedding generation complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate random embeddings based on a template HDF5 file."
    )
    parser.add_argument(
        "--template_h5",
        type=str,
        default="data/processed/sprot_embs/prott5.h5",
        help="Path to the template HDF5 embedding file (e.g., data/processed/sprot_embs/prott5.h5).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/sprot_embs",
        help="Directory to save the generated random embedding files.",
    )
    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=[512, 1024, 2560],
        help="List of embedding dimensions to generate.",
    )

    args = parser.parse_args()

    generate_random_embeddings(args.template_h5, args.output_dir, args.dimensions)
