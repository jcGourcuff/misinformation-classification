import pandas as pd

from ..ipcc.load import get_accurate_quotes
from ..quota_climat import LABEL_MAP, MissInformationLabels, load_quota_climat_dataset


def setup_data_for_multi_cls() -> tuple[pd.DataFrame, pd.DataFrame]:
    true_quotes = get_accurate_quotes()
    false_quotes = load_quota_climat_dataset()[["label", "quote"]]

    # keep same context as in bnry classification
    false_quotes["context_1"] = false_quotes["label"]

    # Remove irelevant columns as we removed them in binary task
    # But we keep the label for multi-class classification
    # to give the opportunity of a 'trash' class for edgy cases
    false_quotes = false_quotes[
        false_quotes["label"] != MissInformationLabels.NOT_RELEVANT
    ]
    false_quotes["label"] = false_quotes["label"].map(LABEL_MAP)
    false_quotes["context_2"] = "N/A"

    true_quotes["label"] = "accurate statement"
    true_quotes["context_1"] = true_quotes["personae"]
    true_quotes["context_2"] = true_quotes["emotion"]
    true_quotes = true_quotes[["label", "quote", "context_1", "context_2"]]

    return true_quotes, false_quotes
