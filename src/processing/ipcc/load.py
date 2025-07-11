import json
from os.path import join

import pandas as pd

from src.conf import DATA_SYNTHESIS_FILE_NAME, IPCC_DIR
from src.mistral.inference.batch import get_batch_job_result


def get_accurate_quotes(reload: bool = False) -> pd.DataFrame:
    if reload:
        dataset: dict[str, list[str]] = {
            "personae": [],
            "emotion": [],
            "quote": [],
        }
        for item in get_batch_job_result(file_name=DATA_SYNTHESIS_FILE_NAME):
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
        dataset_as_df.to_csv(
            join(IPCC_DIR, f"{DATA_SYNTHESIS_FILE_NAME}.csv"), index=False
        )
    return pd.read_csv(join(IPCC_DIR, f"{DATA_SYNTHESIS_FILE_NAME}.csv"))
