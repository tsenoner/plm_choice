from dataclasses import dataclass
from pathlib import Path


@dataclass
class PathsConfig:
    data_dir: Path = Path("data/swissprot")  # Base directory for data
    embeddings_file: Path = Path(
        "embeddings/prott5.h5"
    )  # Relative to data_dir
    # Subdirectory within data_dir containing train/val/test.csv
    csv_subdir: str = "training"
    output_dir: Path = Path("models/runs")  # Base output directory for runs


@dataclass
class TrainConfig:
    param_name: str = "fident"
    seed: int = 42
    hidden_size: int = 64
    learning_rate: float = 0.001
    batch_size: int = 1024
    max_epochs: int = 100
    early_stopping_patience: int = 5


@dataclass
class ProjectConfig:
    paths: PathsConfig = PathsConfig()
    training: TrainConfig = TrainConfig()


def load_config() -> ProjectConfig:
    # In the future, this could load from YAML or parse args,
    # but for now, it just returns the default.
    return ProjectConfig()
