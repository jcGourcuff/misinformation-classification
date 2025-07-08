from enum import StrEnum
from os import makedirs
from os.path import join

import pandas as pd
from datasets import load_dataset

QUOTA_CLIMAT_DATASET = "QuotaClimat/frugalaichallenge-text-train"


class MissInformationLabels(StrEnum):
    NOT_RELEVANT = "0_not_relevant"
    PROPONENTS_BIASED = "6_proponents_biased"
    NOT_BAD = "3_not_bad"
    NOT_HUMAN = "2_not_human"
    NOT_HAPPENING = "1_not_happening"
    SOLUTIONS_HARMFUL_UNNECESSARY = "4_solutions_harmful_unnecessary"
    SCIENCE_UNRELIABLE = "5_science_unreliable"
    FOSSIL_FUELS_NEEDED = "7_fossil_fuels_needed"


def load_quota_climat_dataset(work_dir: str, reload: bool = False) -> pd.DataFrame:
    makedirs(work_dir, exist_ok=True)

    if reload:
        dataset = load_dataset(QUOTA_CLIMAT_DATASET, cache_dir=work_dir)

        full_dataset = pd.concat(
            [dataset["train"].to_pandas(), dataset["test"].to_pandas()],
            ignore_index=True,
        )
        full_dataset.to_csv(join(work_dir, "quota_climat_dataset.csv"), index=False)

    try:
        return pd.read_csv(join(work_dir, "quota_climat_dataset.csv"))
    except FileNotFoundError as err:
        raise FileNotFoundError(
            "Dataset not found. Please run the script with reload=True to download it."
        ) from err
