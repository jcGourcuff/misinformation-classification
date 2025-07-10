import json
from os import makedirs
from os.path import dirname, join

import pandas as pd

from src.classification.multiclass._build_data_set import setup_data_for_multi_cls
from src.inference._batch_inference import BatchedPrompt, BatchRequest
from src.processing.reformat_data import reformat_jsonl
from src.serializer import ReferenceSerializer

PROCCESSING_DIR = "./data/processed"
MULTI_CLS_DATASET_DIR = join(PROCCESSING_DIR, "./multi_cls_dataset")
makedirs(MULTI_CLS_DATASET_DIR, exist_ok=True)
FINETUNING_MULTI_CLS_PROMPT_PATH = join(dirname(__file__), "prompt.txt")


def generate_file_for_multi_cls_finetune():
    _generate_file_for_multi_cls_finetune(which="train")
    _generate_file_for_multi_cls_finetune(which="validation")


def _generate_file_for_multi_cls_finetune(which: str = "train"):

    file_name = f"fine_tune_multi_cls_dataset_{which}"
    file_path = BatchRequest.file_path(file_name)

    dataset = ReferenceSerializer.load(
        join(MULTI_CLS_DATASET_DIR, "fine_tune_multi_cls_dataset.pkl.gz")
    )[which]

    prompt = open(FINETUNING_MULTI_CLS_PROMPT_PATH, encoding="utf-8").read()
    batch_elems = []
    for quote, label in zip(dataset["quote"], dataset["label"]):
        batch_elems.append(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": prompt.format(quote=quote)},
                        {"role": "assistant", "content": label},
                    ]
                }
            )
        )
    with open(file_path, "wb") as file:
        for line in batch_elems:
            file.write((f"{line}\n".encode("utf-8")))
    reformat_jsonl(file_path)
    print(f"Generated {len(batch_elems)} prompts for {file_name}")


def build_finetune_multi_cls_dataset():
    true_quotes, false_quotes = setup_data_for_multi_cls()

    dataset = pd.concat([true_quotes, false_quotes], axis=0, ignore_index=True)

    train_sets = []
    validation_sets = []
    for label in dataset["label"].unique():
        sub_df = dataset[dataset["label"] == label].copy().reset_index(drop=True)
        train_set = sub_df.sample(frac=0.8, random_state=42, replace=False)
        validation_set = sub_df.drop(train_set.index)
        train_sets.append(train_set)
        validation_sets.append(validation_set)

    # Here only for evluation purposes
    ReferenceSerializer.dump(
        {
            "train": pd.concat(train_sets, ignore_index=True),
            "validation": pd.concat(validation_sets, ignore_index=True),
        },
        join(MULTI_CLS_DATASET_DIR, "fine_tune_multi_cls_dataset.pkl.gz"),
    )
