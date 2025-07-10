import pandas as pd

from src.conf import MULTI_CLS_DATASET_FILE
from src.utils import ReferenceSerializer

from ._utils import setup_data_for_multi_cls


def build_multi_cls_dataset():

    true_quotes, false_quotes = setup_data_for_multi_cls()

    dataset = pd.concat(
        [true_quotes, false_quotes],
        axis=0,
        ignore_index=True,
    ).reset_index(drop=True)

    # Here only for evluation purposes
    ReferenceSerializer.dump(
        dataset,
        MULTI_CLS_DATASET_FILE,
    )
