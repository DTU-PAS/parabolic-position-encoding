from pathlib import Path


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def get_datasets_dir(raise_missing: bool = True) -> Path:
    root = get_project_root()

    path = root / "datasets"
    if raise_missing and not path.exists():
        raise ValueError(f"Directory {path} does not exist, please create it.")

    return path


def get_dataset_dir(name: str, raise_missing: bool = True) -> Path:
    path = get_datasets_dir(raise_missing=raise_missing) / name
    if raise_missing and not path.exists():
        raise ValueError(f"Directory {path} does not exist, please create it.")

    return path


def get_experiments_dir(raise_missing: bool = True) -> Path:
    root = get_project_root()

    path = root / "experiments"
    if raise_missing and not path.exists():
        raise ValueError(f"Directory {path} does not exist, please create it.")

    return path


def get_experiment_dir(name: str, raise_missing: bool = True):
    path = get_experiments_dir(raise_missing=raise_missing) / name

    if raise_missing and not path.exists():
        raise ValueError(f"Directory {path} does not exist, please create it.")

    return path
