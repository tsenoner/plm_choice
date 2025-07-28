"""
Centralized experiment management for path handling, completion checking, and state management.
Reduces code duplication across train.py, run_experiments.py, and evaluate.py.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


@dataclass
class ExperimentPaths:
    """Structured container for all experiment-related paths."""

    project_root: Path
    data_dir: Path
    embedding_file: Path
    experiment_dir: Path

    # Data files
    train_file: Path
    val_file: Path
    test_file: Optional[Path]

    # Run artifacts
    checkpoints_dir: Path
    completion_marker: Path
    last_checkpoint: Path


class ExperimentManager:
    """Centralized manager for experiment paths, state, and operations."""

    def __init__(
        self,
        dataset_dir: Path,
        embedding_name: str,
        model_type: str,
        param_name: str,
        models_base_dir: Path,
    ):
        """
        Initialize experiment manager with enforced directory structure.

        Args:
            dataset_dir: Dataset directory containing 'embeddings/' and 'sets/' subdirectories
                        (e.g., 'data/processed/sprot_pre2024')
            embedding_name: Name of the embedding file without extension (e.g., 'esm1b')
            model_type: Type of model (fnn, linear, etc.)
            param_name: Target parameter name
            models_base_dir: Base directory for models (e.g., 'models')
        """
        self.dataset_dir = Path(dataset_dir)
        self.embedding_name = embedding_name
        self.model_type = model_type
        self.param_name = param_name

        # Enforce directory structure
        self.data_dir = self.dataset_dir / "sets"
        self.embeddings_dir = self.dataset_dir / "embeddings"
        self.embedding_file = self.embeddings_dir / f"{embedding_name}.h5"

        # Derive dataset name from directory
        self.dataset_name = self.dataset_dir.name

        # Set models base directory (no inference, always provided)
        self.models_base_dir = Path(models_base_dir)

        # Derive experiment directory
        self.experiment_dir = (
            self.models_base_dir
            / self.dataset_name
            / model_type
            / param_name
            / embedding_name
        )

    def create_experiment_paths(
        self, project_root: Optional[Path] = None
    ) -> ExperimentPaths:
        """
        Create or resolve all experiment paths.

        Args:
            project_root: Project root path (inferred if not provided)

        Returns:
            ExperimentPaths object with all resolved paths
        """
        if project_root is None:
            # Infer project root (3 levels up from src/shared/)
            project_root = Path(__file__).parent.parent.parent

        # Validate enforced directory structure
        if not self.dataset_dir.exists():
            raise NotADirectoryError(f"Dataset directory not found: {self.dataset_dir}")
        if not self.data_dir.exists():
            raise NotADirectoryError(f"Sets directory not found: {self.data_dir}")
        if not self.embeddings_dir.exists():
            raise NotADirectoryError(
                f"Embeddings directory not found: {self.embeddings_dir}"
            )
        if not self.embedding_file.exists():
            raise FileNotFoundError(f"Embedding file not found: {self.embedding_file}")

        # Validate data files
        train_file = self.data_dir / "train.parquet"
        val_file = self.data_dir / "val.parquet"
        test_file = self.data_dir / "test.parquet"

        if not train_file.exists():
            raise FileNotFoundError(f"Train file not found: {train_file}")
        if not val_file.exists():
            raise FileNotFoundError(f"Validation file not found: {val_file}")
        if not test_file.exists():
            print(f"Warning: Test file not found: {test_file}")
            test_file = None

        # Create directory structure
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = self.experiment_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        print(f"Using experiment directory: {self.experiment_dir}")

        return ExperimentPaths(
            project_root=project_root,
            data_dir=self.data_dir,
            embedding_file=self.embedding_file,
            experiment_dir=self.experiment_dir,
            train_file=train_file,
            val_file=val_file,
            test_file=test_file,
            checkpoints_dir=checkpoints_dir,
            completion_marker=self.experiment_dir / "training_complete.txt",
            last_checkpoint=checkpoints_dir / "last.ckpt",
        )

    def check_experiment_status(self) -> Tuple[str, Path]:
        """
        Check the status of the experiment.

        Returns:
            Tuple of (status, experiment_dir) where status is one of:
            - "not_started": No experiment directory exists
            - "completed": Training completed successfully
            - "interrupted": Training started but not completed (has checkpoints)
            - "empty": Experiment directory exists but no meaningful content
        """
        if not self.experiment_dir.exists():
            return "not_started", self.experiment_dir

        return self._check_experiment_status(self.experiment_dir), self.experiment_dir

    def _check_experiment_status(self, experiment_dir: Path) -> str:
        """Check the status of the experiment directory."""
        completion_marker = experiment_dir / "training_complete.txt"
        last_checkpoint = experiment_dir / "checkpoints" / "last.ckpt"

        if completion_marker.exists():
            return "completed"
        elif last_checkpoint.exists():
            return "interrupted"
        else:
            return "empty"

    def find_best_checkpoint(
        self, experiment_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Find the best checkpoint in the experiment directory."""
        if experiment_dir is None:
            experiment_dir = self.experiment_dir

        checkpoints_dir = experiment_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return None

        # Look for best checkpoint (guaranteed by save_top_k=1)
        best_ckpt_files = list(checkpoints_dir.glob("best-*.ckpt"))
        if not best_ckpt_files:
            return None

        if len(best_ckpt_files) > 1:
            print(
                f"Warning: Found {len(best_ckpt_files)} best checkpoints (expected 1)"
            )

        return best_ckpt_files[0]

    def create_completion_marker(
        self, experiment_dir: Path, best_model_path: str, best_score: float
    ):
        """Create a completion marker file."""
        completion_marker = experiment_dir / "training_complete.txt"
        try:
            with open(completion_marker, "w") as f:
                f.write(f"Training completed successfully at {datetime.now()}\n")
                f.write(f"Best model: {best_model_path}\n")
                f.write(f"Final validation loss: {best_score}\n")
            print(f"Created training completion marker: {completion_marker}")
        except Exception as e:
            print(f"Warning: Could not create completion marker: {e}")

    def get_resume_checkpoint_path(
        self, experiment_dir: Optional[Path] = None
    ) -> Optional[Path]:
        """Get the checkpoint path for resuming training."""
        if experiment_dir is None:
            experiment_dir = self.experiment_dir
        last_ckpt = experiment_dir / "checkpoints" / "last.ckpt"
        return last_ckpt if last_ckpt.exists() else None

    @staticmethod
    def from_hparams(hparams: dict, models_base_dir: Path) -> "ExperimentManager":
        """Create ExperimentManager from hyperparameters dictionary."""
        # Extract dataset directory from data_dir in hparams
        data_dir = Path(hparams["data_dir"])

        # If data_dir points to the sets subdirectory, get the parent (dataset directory)
        if data_dir.name == "sets":
            dataset_dir = data_dir.parent
        else:
            dataset_dir = data_dir

        # Extract embedding name from embedding_file path
        embedding_file = Path(hparams["embedding_file"])
        embedding_name = embedding_file.stem

        return ExperimentManager(
            dataset_dir=dataset_dir,
            embedding_name=embedding_name,
            model_type=hparams["model_type"],
            param_name=hparams["param_name"],
            models_base_dir=models_base_dir,
        )
