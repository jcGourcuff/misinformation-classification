from os.path import isfile

import pandas as pd

from src.conf import BINARY_CLS_DATASET_FILE
from src.utils._serializer import ReferenceSerializer

from ..inference.batch import get_batch_job_result
from ._utils import get_result_file_name


def get_binary_cls_result(model: str, reload: bool = False):
    file_name = get_result_file_name(model=model, task="binary_cls")
    if reload or not isfile(file_name):
        dataset: dict = {
            "index": [],
            "predicted_label": [],
        }
        for item in get_batch_job_result(file_name=file_name):
            split_custom_id = item["custom_id"].split("_")
            dataset["index"].append(int(split_custom_id[0]))

            content = item["response"]["body"]["choices"][0]["message"]["content"]
            if content not in ["accurate statement", "misinformation"]:
                raise ValueError(
                    f"Unexpected content: {content}. "
                    "Expected 'accurate statement' or 'misinformation'."
                )
            dataset["predicted_label"].append(content)
        dataset_as_df = pd.DataFrame(dataset).set_index("index")

        # Join to old dataset to get more row context
        original_dataset = ReferenceSerializer.load(BINARY_CLS_DATASET_FILE)
        dataset_as_df = dataset_as_df.join(original_dataset, how="left")

        dataset_as_df.to_csv(file_name, index=False)

    return pd.read_csv(file_name)
