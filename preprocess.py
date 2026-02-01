import tyro

from pape.configs import Dataset
from pape.preprocessing.dvs_gesture import DVSGesturePreprocessor
from pape.preprocessing.gen1 import Gen1Preprocessor


def main(
    dataset: Dataset,
    /,
    limit: int | None = None,
    chunk_duration_ms: int = 250,
    train: bool = True,
    val: bool = True,
    test: bool = True,
):
    match dataset:
        case Dataset.dvsgesture:
            preprocessor = DVSGesturePreprocessor(limit=limit, test=test, train=train)
        case Dataset.gen1:
            preprocessor = Gen1Preprocessor(
                chunk_duration_ms=chunk_duration_ms, limit=limit, test=test, train=train, val=val
            )
        case _:
            raise ValueError(f"Missing preprocessor for dataset: {dataset}")

    preprocessor.preprocess()


tyro.cli(main)
