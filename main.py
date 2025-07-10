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

from src.classification.binary import (
    build_binary_cls_dataset,
    generate_batch_file_for_bin_cls,
    get_binary_cls_result,
)
from src.classification.multiclass._build_data_set import (
    build_multi_cls_dataset,
    get_example_string,
)
from src.classification.multiclass._predict import (
    MULT_CLS_PROMPT_PATH,
    generate_batch_file_for_multi_cls,
    get_multi_cls_result,
)
from src.data_synthesis import (
    generate_batch_file_for_pos_sample_gen,
    generate_postive_data_sample,
)
from src.data_synthesis._load import get_true_quotes
from src.evaluation.explanability import get_breakdown_per_contexts
from src.evaluation.metrics import build_metrics_from_confusion, get_confusion_matrix
from src.fine_tune import build_finetune_multi_cls_dataset
from src.fine_tune._build_data_set import (
    FINETUNING_MULTI_CLS_PROMPT_PATH,
    MULTI_CLS_DATASET_DIR,
    generate_file_for_multi_cls_finetune,
)
from src.fine_tune._run import launch_fine_tune_job
from src.inference import get_batch_job_result, run_batch_mistral
from src.inference._batch_inference import upload_file
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

    file_name = f"binary_cls"
    if build_dataset:
        generate_batch_file_for_bin_cls(file_name)
        upload_file(file_name)
        build_binary_cls_dataset()

    if run:

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


def multiclass_classification_task(
    model: str,
    run: bool = False,
    build_dataset: bool = False,
    validation: bool = False,
    zero_shot: bool = False,
):
    """
    If validation is false, task is ran over the whole dataset.
    Otherwise, we use the finetuning's task validation dataset.
    """
    if build_dataset and not validation:
        build_multi_cls_dataset()

    file_name = f"multi_cls"

    if not validation:
        dataset_info = ReferenceSerializer.load(
            join(MULTI_CLS_DATASET_DIR, "multi_cls_dataset.pkl.gz")
        )
        dataset = dataset_info["dataset"]
    else:
        file_name += "_validation"
        dataset = ReferenceSerializer.load(
            join(MULTI_CLS_DATASET_DIR, "fine_tune_multi_cls_dataset.pkl.gz")
        )["validation"]

    if zero_shot:
        prompt_path = FINETUNING_MULTI_CLS_PROMPT_PATH
        example_string = None
    else:
        prompt_path = MULT_CLS_PROMPT_PATH
        file_name += "_few_shot"
        dataset, example_string = get_example_string(dataset, n_per_class=1)

    if build_dataset:
        generate_batch_file_for_multi_cls(
            file_name,
            dataset=dataset,
            prompt_path=prompt_path,
            example_string=example_string,
        )
        upload_file(file_name)
        return

    if run:
        run_batch_mistral(
            file_name=file_name,
            model=model,
            mode="chat",
            job_type="multi-cls",
        )
        return

    result = get_multi_cls_result(
        file_name=file_name, model=model, original_dataset=dataset, reload=True
    ).fillna("N/A")

    confusion_matrix = get_confusion_matrix(result)
    print(confusion_matrix)
    metrics = build_metrics_from_confusion(confusion_matrix)
    print(metrics)

    ## THEN switch back to binary classification


def finetune_multi_cls_task(
    model: str,
    version: str,
    epochs: int,
    run: bool = False,
    build_dataset: bool = False,
):

    file_name_train = "fine_tune_multi_cls_dataset_train"
    file_name_validation = "fine_tune_multi_cls_dataset_validation"
    # upload_file(file_name_train)

    if build_dataset:
        build_finetune_multi_cls_dataset()
        # generate_file_for_multi_cls_finetune()
        # upload_file(file_name_train, purpose="fine-tune")
        # upload_file(file_name_validation, purpose="fine-tune")

    if run:
        launch_fine_tune_job(
            file_name_train=file_name_train,
            file_name_validation=file_name_validation,
            model=model,
            version=version,
            epochs=epochs,
        )
        return


def main():
    model = "ministral-3b-latest"
    # model = "ministral-8b-latest"
    # model = "mistral-small-latest"
    # model = "ministral-3b-latest_fine_tuned_{version}"

    multiclass_classification_task(
        model=model,
        build_dataset=False,
        run=False,
        validation=True,
        zero_shot=False,
    )

    finetune_multi_cls_task(
        model=model, build_dataset=False, run=False, version="v3", epochs=0
    )

    # client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    # pprint.pprint(client.fine_tuning.jobs.list(status="QUEUED"))


if __name__ == "__main__":
    main()
