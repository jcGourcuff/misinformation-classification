import json
import os
from os.path import join

import pandas as pd

from src.inference._batch_inference import get_batch_job_result

RAW_DATA_DIR = "./data/raw"
SYNTHETIC_DATA_DIR = join(RAW_DATA_DIR, "synthetic_data")
os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)


def get_true_quotes(
    file_name: str = "data_synthesis_v0", reload: bool = False
) -> pd.DataFrame:
    if reload:
        dataset = {
            "personae": [],
            "emotion": [],
            "quote": [],
        }
        for item in get_batch_job_result(file_name=file_name):
            split_custom_id = item["custom_id"].split("_")
            # hotfix TODO
            if "climate_enthusiast" in item["custom_id"]:
                dataset["personae"].append("climate_enthusiast")
                dataset["emotion"].append(split_custom_id[2])
            else:
                dataset["personae"].append(split_custom_id[0])
                dataset["emotion"].append(split_custom_id[1])
            content = (
                item["response"]["body"]["choices"][0]["message"]["content"]
                .replace("\\", "")
                .strip("```json")
                .strip("```")
            )
            dataset["quote"].append(json.loads(content)["text"])
        dataset_as_df = pd.DataFrame(dataset)
        dataset_as_df.to_csv(join(SYNTHETIC_DATA_DIR, f"{file_name}.csv"), index=False)
    return pd.read_csv(join(SYNTHETIC_DATA_DIR, f"{file_name}.csv"))
