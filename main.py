import json
import os
import pprint
import re
from datetime import datetime
from os import getenv
from os.path import join
from typing import Literal

import numpy as np
import pandas as pd
from mistralai import Mistral
from sklearn.metrics import confusion_matrix

from src.binary_cls._build_data_set import (
    BINARY_CLS_DATASET_DIR,
    build_binary_cls_dataset,
)
from src.binary_cls._predict import (
    generate_batch_file_for_bin_cls,
    get_binary_cls_result,
)
from src.data_synthesis import (
    generate_batch_file_for_pos_sample_gen,
    generate_postive_data_sample,
)
from src.data_synthesis._load import get_true_quotes
from src.evaluation.explanability import get_breakdown_per_contexts
from src.evaluation.metrics import build_metrics_from_confusion, get_confusion_matrix
from src.inference import get_batch_job_result, run_batch_mistral
from src.processing import load_and_process_ipcc_reports, load_quota_climat_dataset
from src.serializer import ReferenceSerializer

RAW_DATA_DIR = "./data/raw"
IPCC_REPORTS_DIR = join(RAW_DATA_DIR, "IPCC")
SYNTHETIC_DATA_DIR = join(RAW_DATA_DIR, "synthetic_data")
os.makedirs(SYNTHETIC_DATA_DIR, exist_ok=True)
os.makedirs(IPCC_REPORTS_DIR, exist_ok=True)


def load_raw_data():
    load_quota_climat_dataset(work_dir=RAW_DATA_DIR, reload=True)
    load_and_process_ipcc_reports(work_dir=IPCC_REPORTS_DIR, reload=True)


def generate_synthetic_data_from_ipcc_reports(file_name: str = "data_synthesis_v0"):

    text_blocks = load_and_process_ipcc_reports(work_dir=IPCC_REPORTS_DIR)

    generate_batch_file_for_pos_sample_gen(
        ipcc_report_blocks=text_blocks,
        file_name=file_name,
    )

    run_batch_mistral(
        file_name=file_name,
        model="mistral-large-latest",
        mode="chat",
        job_type="data-synthesis",
    )

    get_true_quotes(file_name=file_name)


def binary_classification_task(
    model: str,
    run: bool = False,
    build_dataset: bool = False,
):
    if build_dataset:
        build_binary_cls_dataset()

    file_name = f"binary_cls_{model}"

    if run:
        generate_batch_file_for_bin_cls(file_name)
        run_batch_mistral(
            file_name=file_name,
            model=model,
            mode="chat",
            job_type="binary-cls",
        )
        return

    result = get_binary_cls_result(model=model, reload=False).fillna("N/A")

    confusion_matrix = get_confusion_matrix(result)
    print(confusion_matrix)
    metrics = build_metrics_from_confusion(confusion_matrix)
    print(metrics)

    miss_predictions = result[result["predicted_label"] != result["label"]].copy()

    for breakdown in get_breakdown_per_contexts(miss_predictions):
        print(breakdown)


def main():
    model = "ministral-3b-latest"
    # model = "ministral-8b-latest"
    # model = "mistral-small-latest"


if __name__ == "__main__":
    main()
