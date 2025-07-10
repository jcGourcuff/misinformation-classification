import random

import pandas as pd


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
