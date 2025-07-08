import random
from os import makedirs
from os.path import join

import pandas as pd

from src.data_synthesis import get_true_quotes
from src.processing import load_quota_climat_dataset
from src.processing._quota_climat import MissInformationLabels
from src.serializer import ReferenceSerializer
from src.utils import extract_random_sample, generate_classification_example

RAW_DATA_DIR = "./data/raw"
SYNTHETIC_DATA_DIR = join(RAW_DATA_DIR, "synthetic_data")

PROCCESSING_DIR = "./data/processed"
BINARY_CLS_DATASET_DIR = join(PROCCESSING_DIR, "./binary_cls_dataset")
makedirs(BINARY_CLS_DATASET_DIR, exist_ok=True)

"""
Use metrics that are not sensitive to class imbalance, such as precision, recall, F1-score, and the area under the ROC curve (AUC-ROC), instead of accuracy.
Consider using the confusion matrix to get a detailed view of the model's performance across different classes.
"""


def build_binary_cls_dataset():
    random.seed(42)

    true_quotes = get_true_quotes()
    false_quotes = load_quota_climat_dataset(RAW_DATA_DIR)[["label", "quote"]]

    ## Remove irelevant columns and re-label
    false_quotes = false_quotes[
        false_quotes["label"] != MissInformationLabels.NOT_RELEVANT
    ]
    false_quotes["label"] = "misinformation"

    true_quotes["label"] = "accurate statement"
    true_quotes = true_quotes[["label", "quote"]]

    ## Get one sample per label for prompt
    false_quotes, false_examples = extract_random_sample(false_quotes, size=3)
    true_quotes, true_examples = extract_random_sample(true_quotes, size=3)

    example_string = generate_classification_example(
        class_quote_map={
            "misinformation": false_examples,
            "accurate statement": true_examples,
        }
    )

    # Here only for evluation purposes
    dataset = pd.concat(
        [true_quotes, false_quotes],
        ignore_index=True,
    ).reset_index(drop=True)
    ReferenceSerializer.dump(
        {
            "dataset": dataset,
            "example_string": example_string,
        },
        join(BINARY_CLS_DATASET_DIR, "binary_cls_dataset.pkl.gz"),
    )
