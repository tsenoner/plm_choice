# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on: Mon 17 Oct 2022 19:40:19
Description: extract best predicted PDB models from ColabFold output
Usage:       python get_best_pdbs.py

@author: tsenoner
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import List, Optional


def setup_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract best PDB models from ColabFold output."
    )
    parser.add_argument(
        "-cf",
        "--colabfold-dir",
        dest="colabfold_dir",
        required=True,
        type=Path,
        help="Path to the ColabFold output directory.",
    )
    parser.add_argument(
        "-f",
        "--fasta-file",
        dest="fasta_file",
        required=False,
        type=Path,
        help="Path to FASTA file to extract headers from.",
    )
    return parser.parse_args()


def get_fasta_headers(fasta_file: Path) -> List[str]:
    """Read a FASTA file and return a list of its headers."""
    headers = []
    with fasta_file.open("r") as f:
        for line in f:
            if line.startswith(">"):
                headers.append(line[1:].strip())
    return headers


def sanitize_filename(filename: str) -> str:
    """Remove characters from a string that are not suitable for a filename."""
    return re.sub(r'[\\/*?:"<>|]', "_", filename)


def get_protein_id(pdb_fn: Path) -> str:
    """Extract the protein ID from a PDB filename."""
    protein_id = re.split(r"_(?:scores|unrelaxed)_rank_001", pdb_fn.stem)[0]
    return protein_id


def extract_rank1_models(
    from_dir: Path,
    pdb_dir: Path,
    json_dir: Path,
    fasta_headers: Optional[List[str]] = None,
) -> None:
    """Copy rank_1 PDB and JSON files, renaming them with the protein header."""
    for directory in [pdb_dir, json_dir]:
        directory.mkdir(exist_ok=True)

    rank1_files = from_dir.glob("*rank_001*")
    for file_path in rank1_files:
        protein_id_str = get_protein_id(file_path)
        header = protein_id_str

        if fasta_headers:
            protein_idx = int(protein_id_str)
            header = sanitize_filename(fasta_headers[protein_idx])

        if file_path.suffix == ".pdb":
            dest_dir = pdb_dir
        elif file_path.suffix == ".json":
            dest_dir = json_dir
        else:
            continue

        new_fn = dest_dir / f"{header}{file_path.suffix}"
        if not new_fn.is_file():
            shutil.copy(file_path, new_fn)


def main() -> None:
    """Main function to run the script."""
    args = setup_arguments()
    colabfold_dir = args.colabfold_dir
    fasta_file = args.fasta_file

    if not colabfold_dir.is_dir():
        print(f"Error: Directory not found at {colabfold_dir}")
        return

    predictions_dir = colabfold_dir / "predictions"
    if not predictions_dir.is_dir():
        print(f"Error: 'predictions' directory not found in {colabfold_dir}")
        return

    fasta_headers: Optional[List[str]] = None
    if fasta_file:
        try:
            fasta_headers = get_fasta_headers(fasta_file)
        except FileNotFoundError as e:
            print(f"FASTA file not found at {fasta_file}: {e}")
            fasta_headers = None

    pdb_dir = colabfold_dir / "pdb"
    json_dir = colabfold_dir / "metric"

    extract_rank1_models(
        from_dir=predictions_dir,
        pdb_dir=pdb_dir,
        json_dir=json_dir,
        fasta_headers=fasta_headers,
    )


if __name__ == "__main__":
    main()
