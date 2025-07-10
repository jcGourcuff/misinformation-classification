import random

import pandas as pd


def extract_random_sample(
    data: pd.DataFrame, size: int
) -> tuple[pd.DataFrame, list[str]]:
    data = data.copy().reset_index(drop=True)
    _gen_id = random.sample(list(range(len(data))), size)
    res = [data.loc[i, "quote"] for i in _gen_id]
    data = data.drop(_gen_id, axis=0).reset_index(drop=True)
    return data, res


def generate_classification_example(class_quote_map: dict[str, list[str]]) -> str:
    res = []
    for cls, quotes in class_quote_map.items():
        for quote in quotes:
            res.append(f"Statement: {quote}\nCategory: {cls}")
    random.shuffle(res)
    return "\n".join(res)
