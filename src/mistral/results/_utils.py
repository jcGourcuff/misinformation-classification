from os.path import join

from src.conf import BINARY_CLS_DATASET_DIR, MULTI_CLS_DATASET_DIR

from ..prep import Task


def get_result_file_name(
    model: str,
    task: Task,
) -> str:
    directory = (
        BINARY_CLS_DATASET_DIR if task == "binary_cls" else MULTI_CLS_DATASET_DIR
    )
    file_name = f"{task}_{model}.csv"
    return join(directory, file_name)
