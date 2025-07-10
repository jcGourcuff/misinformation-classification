import pandas as pd

from src.conf import FINETUNE_DATASET_FILE
from src.utils import ReferenceSerializer

from ._utils import setup_data_for_multi_cls


def build_finetune_dataset() -> None:
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
        FINETUNE_DATASET_FILE,
    )
