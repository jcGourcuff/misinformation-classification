from enum import StrEnum
from os.path import isfile, join

import pandas as pd
from datasets import load_dataset

from src.conf import QUOTA_CLIMAT_DATASET, QUOTA_CLIMAT_DIR
from src.utils import logger


class MissInformationLabels(StrEnum):
    NOT_RELEVANT = "0_not_relevant"
    NOT_BAD = "3_not_bad"
    NOT_HUMAN = "2_not_human"
    NOT_HAPPENING = "1_not_happening"
    SOLUTIONS_HARMFUL_UNNECESSARY = "4_solutions_harmful_unnecessary"
    SCIENCE_UNRELIABLE = "5_science_unreliable"
    PROPONENTS_BIASED = "6_proponents_biased"
    FOSSIL_FUELS_NEEDED = "7_fossil_fuels_needed"


LABEL_MAP = {
    "accurate statement": "accurate statement",
    MissInformationLabels.NOT_RELEVANT: "not relevant",
    MissInformationLabels.NOT_HAPPENING: "not happening",
    MissInformationLabels.NOT_HUMAN: "not human",
    MissInformationLabels.NOT_BAD: "not bad",
    MissInformationLabels.SOLUTIONS_HARMFUL_UNNECESSARY: "solutions harmful unnecessary",
    MissInformationLabels.SCIENCE_UNRELIABLE: "science unreliable",
    MissInformationLabels.PROPONENTS_BIASED: "proponents biased",
    MissInformationLabels.FOSSIL_FUELS_NEEDED: "fossil fuels needed",
}


def load_quota_climat_dataset(reload: bool = False) -> pd.DataFrame:
    file_path = join(QUOTA_CLIMAT_DIR, "quota_climat_dataset.csv")
    if reload or not isfile(file_path):

        logger.info(
            "Downloading Hugging Face's %s ",
            QUOTA_CLIMAT_DIR,
        )
        dataset = load_dataset(QUOTA_CLIMAT_DATASET, cache_dir=QUOTA_CLIMAT_DIR)

        full_dataset = pd.concat(
            [dataset["train"].to_pandas(), dataset["test"].to_pandas()],
            ignore_index=True,
        )
        full_dataset.to_csv(file_path, index=False)

    return pd.read_csv(file_path)
