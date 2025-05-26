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
    "esm2_8M": {  # family_key: esm_transformer
        "expected_dim_per_protein": 320,
        "expected_dim_per_residue_axis1": 320,
    },
    "ankh_base": {  # family_key: ankh
        "expected_dim_per_protein": 768,
        "expected_dim_per_residue_axis1": 768,
    },
    "prot_t5_xl_uniref50": {  # family_key: prot_t5
        "expected_dim_per_protein": 1024,
        "expected_dim_per_residue_axis1": 1024,
    },
    # Native ESM models require login. The script itself handles finding tokens
    # from default paths or --token_path. Tests will run them directly.
    "esm3_open_native": {  # family_key: native_esm3
        "expected_dim_per_protein": 1536,
        "expected_dim_per_residue_axis1": 1536,
        "requires_login_setup": True,  # This key can still be informative
        "per_residue_len_adjust": 2,  # Accounts for start/end tokens often included by native ESM
    },
    "esmc_300m_native": {  # family_key: native_esmc
        "expected_dim_per_protein": 960,  # Corrected from 1024
        "expected_dim_per_residue_axis1": 960,  # Corrected from 1024
        "requires_login_setup": True,  # This key can still be informative
        "per_residue_len_adjust": 2,  # Accounts for start/end tokens
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
        group_key = model_key.replace("/", "_")
        assert group_key in hf, f"Model group '{group_key}' not found in HDF5 file."
        model_group = hf[group_key]

        # All valid sequences (9 out of 10) should be processed as default max_seq_len is 2000
        assert len(model_group.keys()) == 9, (
            f"Expected 9 embeddings, found {len(model_group.keys())}"
        )

        assert "seq1_short" in model_group
        assert "seq5_empty_sequence" not in model_group  # Skipped as it's empty
        assert "seq7_potentially_long_for_testing_max_len_30" in model_group

        emb = model_group["seq1_short"][:]
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
        group_key = model_key.replace("/", "_")
        assert group_key in hf
        model_group = hf[group_key]
        assert len(model_group.keys()) == 9

        assert "seq2_standard_length" in model_group
        emb = model_group["seq2_standard_length"][:]
        assert emb.ndim == 2, "Per-residue embedding should be 2D"

        original_seq_len_seq2 = 108
        # Adjust expected length for native ESM models if they include start/end tokens
        len_adjustment = config.get("per_residue_len_adjust", 0)
        expected_len = original_seq_len_seq2 + len_adjustment

        assert emb.shape[0] == expected_len, (
            f"Expected length {expected_len} for seq2 ({model_key}), got {emb.shape[0]}"
        )
        assert emb.shape[1] == config["expected_dim_per_residue_axis1"], (
            f"Incorrect embedding dimension for {model_key}. Expected {config['expected_dim_per_residue_axis1']}, got {emb.shape[1]}"
        )


def test_max_seq_len_skipping(dummy_fasta_file, output_h5_file):
    """Test that max_seq_len correctly skips longer sequences."""
    model_key = "esm2_8M"  # Use a fast model
    max_len = 30
    # Sequences with length > 30: seq2_standard_length (108), seq7 (32)
    # Expected count: 9 (total valid) - 2 (skipped) = 7

    run_script(dummy_fasta_file, model_key, output_h5_file, max_seq_len=max_len)

    assert output_h5_file.exists()
    with h5py.File(output_h5_file, "r") as hf:
        group_key = model_key.replace("/", "_")
        assert group_key in hf
        model_group = hf[group_key]

        assert len(model_group.keys()) == 7, (
            f"Expected 7 embeddings with max_len={max_len}, found {len(model_group.keys())}"
        )
        assert "seq2_standard_length" not in model_group
        assert "seq7_potentially_long_for_testing_max_len_30" not in model_group
        assert "seq1_short" in model_group


def test_append_to_hdf5(dummy_fasta_file, output_h5_file):
    """Test that running the script for different models appends to the same HDF5 file."""
    model1_key = "esm2_8M"
    # Choose a different family model to ensure distinct group creation
    model2_key = "ankh_base"

    run_script(dummy_fasta_file, model1_key, output_h5_file)
    run_script(dummy_fasta_file, model2_key, output_h5_file)  # Append with second model

    assert output_h5_file.exists()
    with h5py.File(output_h5_file, "r") as hf:
        group1_key = model1_key.replace("/", "_")
        group2_key = model2_key.replace("/", "_")
        assert group1_key in hf, f"Group for model {model1_key} not found."
        assert group2_key in hf, f"Group for model {model2_key} not found."
        assert len(hf[group1_key].keys()) == 9, (
            f"Incorrect embed count for {model1_key}"
        )
        assert len(hf[group2_key].keys()) == 9, (
            f"Incorrect embed count for {model2_key}"
        )


def test_weights_dir_usage(dummy_fasta_file, output_h5_file, tmp_path_factory):
    """Test that the --weights_dir argument correctly sets the HF cache directory."""
    model_key = "esm2_8M"  # Use a small, fast-downloading model
    custom_weights_dir = tmp_path_factory.mktemp("custom_hf_cache")

    run_script(
        dummy_fasta_file,
        model_key,
        output_h5_file,  # Needs an output file, though we don't check its content here
        weights_dir=custom_weights_dir,
    )

    # Check if the model was downloaded into the custom_weights_dir.
    # Hugging Face typically creates a structure like: <cache_dir>/models--<org>--<model_name>
    # For facebook/esm2_t6_8M_UR50D, this would be models--facebook--esm2_t6_8M_UR50D
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
