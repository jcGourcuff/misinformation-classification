from os.path import join
from typing import Literal

from src.conf import BINARY_CLS_DATASET_DIR, MULTI_CLS_DATASET_DIR


def get_result_file_name(
    model: str,
    task: Literal[
        "binary_cls",
        "multi_cls_global",
        "multi_cls_validation_zero_shot",
        "multi_cls_validation_few_shots",
    ],
) -> str:
    directory = (
        BINARY_CLS_DATASET_DIR if task == "binary_cls" else MULTI_CLS_DATASET_DIR
    )
    file_name = f"{task}_{model}.csv"
    return join(directory, file_name)
