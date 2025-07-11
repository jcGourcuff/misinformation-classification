import random
from typing import Literal, cast

import pandas as pd

Task = Literal[
    "binary_cls",
    "multi_cls_global",
    "multi_cls_validation_zero_shot",
    "multi_cls_validation_few_shots",
]


def get_task_full_name(
    task: Literal["binary_cls", "multi_cls"],
    eval_set: Literal["global", "validation"],
    few_shot: bool,
) -> Task:
    if task == "binary_cls":
        return "binary_cls"
    file_name = f"{task}_{eval_set}"

    if few_shot:
        file_name += "_few_shots"
    else:
        file_name += "_zero_shot"

    return cast(Task, file_name)


def get_example_string(
    dataset: pd.DataFrame, n_per_class: int
) -> tuple[pd.DataFrame, str]:
    random.seed(42)
    new_dset = []
    class_quote_map = {}
    for label in dataset["label"].unique():
        sub_dset, label_examples = _extract_random_sample(
            dataset[dataset["label"] == label], size=n_per_class
        )
        class_quote_map[label] = label_examples
        new_dset.append(sub_dset)
    dataset = pd.concat(
        new_dset,
        ignore_index=False,
    )

    example_string = _generate_classification_example(class_quote_map=class_quote_map)

    return dataset, example_string


def _generate_classification_example(class_quote_map: dict[str, list[str]]) -> str:
    res = []
    for cls, quotes in class_quote_map.items():
        for quote in quotes:
            res.append(f"Statement: {quote}\nCategory: {cls}")
    random.shuffle(res)
    return "\n".join(res)


def _extract_random_sample(
    data: pd.DataFrame, size: int
) -> tuple[pd.DataFrame, list[str]]:
    data = data.copy()
    _gen_id = random.sample(list(data.index), size)
    res = [data.loc[i, "quote"] for i in _gen_id]
    data = data.drop(_gen_id, axis=0)
    return data, res
