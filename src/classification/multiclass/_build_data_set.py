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
MULTI_CLS_DATASET_DIR = join(PROCCESSING_DIR, "./multi_cls_dataset")
makedirs(MULTI_CLS_DATASET_DIR, exist_ok=True)

LABEL_MAP = {
    "0_not_relevant": "not relevant",
    "1_not_happening": "not happening",
    "2_not_human": "not human",
    "3_not_bad": "not bad",
    "4_solutions_harmful_unnecessary": "solutions harmful unnecessary",
    "5_science_unreliable": "science unreliable",
    "6_proponents_biased": "proponents biased",
    "7_fossil_fuels_needed": "fossil fuels needed",
}


def build_multi_cls_dataset():
    random.seed(42)

    true_quotes, false_quotes = setup_data_for_multi_cls()

    ## Get one sample per label for prompt
    dataset, true_examples = extract_random_sample(true_quotes, size=1)
    class_quote_map = {
        "accurate statement": true_examples,
    }

    for label in false_quotes["label"].unique():
        label_quotes, label_examples = extract_random_sample(
            false_quotes[false_quotes["label"] == label], size=1
        )
        class_quote_map[label] = label_examples
        dataset = pd.concat(
            [dataset, label_quotes],
            ignore_index=True,
        )
    dataset = dataset.reset_index(drop=True)
    example_string = generate_classification_example(class_quote_map=class_quote_map)

    # Here only for evluation purposes
    ReferenceSerializer.dump(
        {
            "dataset": dataset,
            "example_string": example_string,
        },
        join(MULTI_CLS_DATASET_DIR, "multi_cls_dataset.pkl.gz"),
    )


def setup_data_for_multi_cls() -> tuple[pd.DataFrame, pd.DataFrame]:
    true_quotes = get_true_quotes()
    false_quotes = load_quota_climat_dataset(RAW_DATA_DIR)[["label", "quote"]]

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
