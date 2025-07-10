from os import makedirs
from os.path import join

import pandas as pd

from src.classification.multiclass._build_data_set import setup_data_for_multi_cls
from src.serializer import ReferenceSerializer

PROCCESSING_DIR = "./data/processed"
MULTI_CLS_DATASET_DIR = join(PROCCESSING_DIR, "./multi_cls_dataset")
makedirs(MULTI_CLS_DATASET_DIR, exist_ok=True)


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
