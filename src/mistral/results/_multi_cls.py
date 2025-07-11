from os.path import isfile, join

import pandas as pd

from src.conf import (
    FINETUNE_DATASET_FILE,
    MULTI_CLS_DATASET_DIR,
    MULTI_CLS_DATASET_FILE,
)
from src.mistral.inference.batch import get_batch_job_result
from src.mistral.prep import Task
from src.processing.quota_climat import LABEL_MAP
from src.utils import logger

from ._utils import get_result_file_name


def get_multi_cls_result(
    task: Task,
    model: str,
    reload: bool = False,
):
    if task == "binary_cls":
        raise ValueError("This function is for multi-class classification tasks only.")
    if task == "multi_cls_global":
        original_dataset = pd.read_csv(MULTI_CLS_DATASET_FILE)
    else:
        original_dataset = pd.read_csv(FINETUNE_DATASET_FILE)["validation"]

    file_name = get_result_file_name(model=model, task=task)
    if reload or not isfile(file_name):
        dataset: dict = {
            "index": [],
            "predicted_label": [],
        }
        for item in get_batch_job_result(file_name=file_name):
            content = item["response"]["body"]["choices"][0]["message"]["content"]
            authorized_labels = list(LABEL_MAP.values())
            if content not in authorized_labels:
                logger.warning("Invalid output: %s.", content)
                continue

            dataset["predicted_label"].append(content)
            split_custom_id = item["custom_id"].split("_")
            dataset["index"].append(int(split_custom_id[0]))
        dataset_as_df = pd.DataFrame(dataset).set_index("index")

        # Join to old dataset
        dataset_as_df = dataset_as_df.join(original_dataset, how="left")

        dataset_as_df.to_csv(
            join(MULTI_CLS_DATASET_DIR, f"{file_name}.csv"), index=False
        )

    return pd.read_csv(join(MULTI_CLS_DATASET_DIR, f"{file_name}.csv"))
