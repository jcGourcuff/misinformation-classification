import pandas as pd

from src.conf import BINARY_CLS_DATASET_FILE
from src.utils import ReferenceSerializer

from ..ipcc.load import get_accurate_quotes
from ..quota_climat import MissInformationLabels, load_quota_climat_dataset


def build_bin_cls_dataset() -> None:
    """
    Only for evaluation purposes.
    """
    true_quotes = get_accurate_quotes()
    false_quotes = load_quota_climat_dataset()[["label", "quote"]]

    ## Remove irelevant columns and re-label
    false_quotes = false_quotes[
        false_quotes["label"] != MissInformationLabels.NOT_RELEVANT
    ]
    false_quotes["context_1"] = false_quotes["label"]
    false_quotes["context_2"] = "N/A"
    false_quotes["label"] = "misinformation"

    true_quotes["label"] = "accurate statement"
    true_quotes["context_1"] = true_quotes["personae"]
    true_quotes["context_2"] = true_quotes["emotion"]
    true_quotes = true_quotes[["label", "quote", "context_1", "context_2"]]

    # Here only for evluation purposes
    dataset = pd.concat(
        [true_quotes, false_quotes],
        ignore_index=True,
    ).reset_index(drop=True)
    ReferenceSerializer.dump(dataset, BINARY_CLS_DATASET_FILE)
