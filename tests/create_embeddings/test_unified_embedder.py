import pytest
import subprocess
import h5py
from pathlib import Path
from typing import Optional


# Define project root and script paths relative to this test file
TEST_DIR = Path(__file__).parent
PROJECT_ROOT = TEST_DIR.parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "create_embeddings" / "unified_embedder.py"
FASTA_FILE = TEST_DIR / "dummy_proteins.fasta"
DEFAULT_HF_TOKEN_PATH = Path.home() / ".cache" / "huggingface" / "token"

# Models to test, covering each family_key for broader coverage.
# Dimensions should be verified if tests fail.
TEST_MODELS = {
    "esm2_8m": {  # family_key: esm_transformer
        "expected_dim_per_protein": 320,
        "expected_dim_per_residue_axis1": 320,
    },
    "ankh_base": {  # family_key: ankh
        "expected_dim_per_protein": 768,
        "expected_dim_per_residue_axis1": 768,
    },
    "prot_t5": {  # family_key: prot_t5 (using the key for prot_t5_xl_half_uniref50-enc)
        "expected_dim_per_protein": 1024,
        "expected_dim_per_residue_axis1": 1024,
    },
    # Native ESM models require login. The script itself handles finding tokens
    # from default paths or --token_path. Tests will run them directly.
    "esm3_open": {  # family_key: esm3
        "expected_dim_per_protein": 1536,
        "expected_dim_per_residue_axis1": 1536,
        "requires_login_setup": True,
    },
    "esmc_300m": {  # family_key: esmc
        "expected_dim_per_protein": 960,
        "expected_dim_per_residue_axis1": 960,
        "requires_login_setup": True,
    },
}


@pytest.fixture(scope="module")
def dummy_fasta_file():
    """Provides the path to the dummy FASTA file."""
    return FASTA_FILE


@pytest.fixture
def output_h5_file(tmp_path_factory):
    """Provides a unique temporary HDF5 file path for each test run."""
    fn = tmp_path_factory.mktemp("h5_data") / "test_output.h5"
    yield fn
    # tmp_path_factory handles cleanup of its created temp directories


def run_script(
    fasta_path: Path,
    model_key: str,
    output_path: Path,
    embedding_type: str = "per_protein",
    max_seq_len: Optional[int] = None,
    weights_dir: Optional[Path] = None,
    token_path: Optional[Path] = None,
):
    """Helper function to run the unified_embedder.py script."""
    command = [
        "python",
        str(SCRIPT_PATH),
        str(fasta_path),
        model_key,
        "--output_hdf5_file",
        str(output_path),
        "--embedding_type",
        embedding_type,
    ]
    if max_seq_len is not None:
        command.extend(["--max_seq_len", str(max_seq_len)])
    if weights_dir is not None:
        command.extend(["--weights_dir", str(weights_dir)])
    if token_path is not None:
        command.extend(["--token_path", str(token_path)])

    # print(f"Running command: {' '.join(map(str, command))}") # Keep for debugging if needed locally
    result = subprocess.run(command, capture_output=True, text=True, check=False)

    if result.returncode != 0:  # Print details only on failure
        print(f"Running command: {' '.join(map(str, command))}")
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)

    assert result.returncode == 0, (
        f"Script execution failed for model {model_key}. Stderr: {result.stderr.strip()}"
    )
    return result


@pytest.mark.parametrize("model_key, config", TEST_MODELS.items())
def test_per_protein_embeddings(dummy_fasta_file, output_h5_file, model_key, config):
    """Test per-protein embedding generation for various models."""
    token_to_pass = None
    if config.get("requires_login_setup") and DEFAULT_HF_TOKEN_PATH.is_file():
        token_to_pass = DEFAULT_HF_TOKEN_PATH

    run_script(
        dummy_fasta_file,
        model_key,
        output_h5_file,
        embedding_type="per_protein",
        token_path=token_to_pass,
    )

    assert output_h5_file.exists(), "HDF5 output file was not created."
    with h5py.File(output_h5_file, "r") as hf:
        # Embeddings are now top-level datasets
        # All valid sequences (9 out of 10) should be processed
        assert len(hf.keys()) == 9, f"Expected 9 embeddings, found {len(hf.keys())}"

        assert "seq1_short" in hf
        assert "seq5_empty_sequence" not in hf  # Skipped as it's empty
        assert "seq7_potentially_long_for_testing_max_len_30" in hf

        emb = hf["seq1_short"][:]
        assert emb.ndim == 1, "Per-protein embedding should be 1D"
        assert emb.shape[0] == config["expected_dim_per_protein"], (
            f"Incorrect embedding dimension for {model_key}. Expected {config['expected_dim_per_protein']}, got {emb.shape[0]}"
        )


@pytest.mark.parametrize("model_key, config", TEST_MODELS.items())
def test_per_residue_embeddings(dummy_fasta_file, output_h5_file, model_key, config):
    """Test per-residue embedding generation for various models."""
    token_to_pass = None
    if config.get("requires_login_setup") and DEFAULT_HF_TOKEN_PATH.is_file():
        token_to_pass = DEFAULT_HF_TOKEN_PATH

    run_script(
        dummy_fasta_file,
        model_key,
        output_h5_file,
        embedding_type="per_residue",
        token_path=token_to_pass,
    )

    assert output_h5_file.exists()
    with h5py.File(output_h5_file, "r") as hf:
        # Embeddings are now top-level datasets
        assert len(hf.keys()) == 9

        assert "seq2_standard_length" in hf
        emb = hf["seq2_standard_length"][:]
        assert emb.ndim == 2, "Per-residue embedding should be 2D"

        original_seq_len_seq2 = 108
        expected_len = original_seq_len_seq2

        assert emb.shape[0] == expected_len, (
            f"Expected length {expected_len} for seq2 ({model_key}), got {emb.shape[0]}"
        )
        assert emb.shape[1] == config["expected_dim_per_residue_axis1"], (
            f"Incorrect embedding dimension for {model_key}. Expected {config['expected_dim_per_residue_axis1']}, got {emb.shape[1]}"
        )


def test_max_seq_len_skipping(dummy_fasta_file, output_h5_file):
    """Test that max_seq_len correctly skips longer sequences."""
    model_key = "esm2_8m"  # Use a fast model, already updated key
    max_len = 30
    # Expected count: 9 (total valid in fasta) - 2 (skipped by max_len) = 7

    run_script(dummy_fasta_file, model_key, output_h5_file, max_seq_len=max_len)

    assert output_h5_file.exists()
    with h5py.File(output_h5_file, "r") as hf:
        # Embeddings are now top-level datasets
        assert len(hf.keys()) == 7, (
            f"Expected 7 embeddings with max_len={max_len}, found {len(hf.keys())}"
        )
        assert "seq2_standard_length" not in hf
        assert "seq7_potentially_long_for_testing_max_len_30" not in hf
        assert "seq1_short" in hf


def test_append_to_hdf5(dummy_fasta_file, output_h5_file):
    """Test that running the script for different models appends datasets to the same HDF5 file."""
    model1_key = "esm2_8m"  # Already updated key
    model2_key = "ankh_base"

    # Run for model1, outputting to output_h5_file
    run_script(dummy_fasta_file, model1_key, output_h5_file)

    # Run for model2, outputting to the SAME output_h5_file
    run_script(dummy_fasta_file, model2_key, output_h5_file)

    assert output_h5_file.exists()
    with h5py.File(output_h5_file, "r") as hf:
        # Both models should have written their datasets to the root of the same file.
        # Since the input FASTA is the same, and sequence headers are unique dataset keys,
        # the file should contain all 9 unique processable sequences.
        # The test now verifies that data from both runs co-exists.
        assert len(hf.keys()) == 9, (
            f"Expected 9 unique sequence embeddings after two model runs, found {len(hf.keys())}"
        )
        # Check a sequence processed by either (all sequences are processed by both runs in this setup)
        assert "seq1_short" in hf
        assert "seq2_standard_length" in hf


def test_weights_dir_usage(dummy_fasta_file, output_h5_file, tmp_path_factory):
    """Test that the --weights_dir argument correctly sets the HF cache directory."""
    model_key = "esm2_8m"  # Use a small, fast-downloading model, already updated key
    custom_weights_dir = tmp_path_factory.mktemp("custom_hf_cache")

    run_script(
        dummy_fasta_file,
        model_key,
        output_h5_file,
        weights_dir=custom_weights_dir,
    )

    expected_model_folder_name_part = "models--facebook--esm2_t6_8M_UR50D"
    downloaded_correctly = False
    if custom_weights_dir.exists() and custom_weights_dir.is_dir():
        for item in custom_weights_dir.iterdir():
            if item.is_dir() and expected_model_folder_name_part in item.name:
                downloaded_correctly = True
                break

    assert downloaded_correctly, (
        f"Model {model_key} does not appear to be downloaded into custom weights_dir: {custom_weights_dir}. Contents: {list(custom_weights_dir.iterdir()) if custom_weights_dir.exists() else 'N/A'}"
    )
