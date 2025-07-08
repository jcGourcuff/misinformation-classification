import os
import random

import pandas as pd


def load_api_key(file_path: str = "./env/MISTRAL_API_KEY") -> None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            api_key = file.read().strip()
        os.environ["MISTRAL_API_KEY"] = api_key
    except FileNotFoundError as err:
        raise FileNotFoundError(f"API key file not found: {file_path}") from err


def extract_random_sample(
    data: pd.DataFrame, size: int
) -> tuple[pd.DataFrame, list[str]]:
    data = data.reset_index(drop=True)
    _gen_id = random.choice(list(range(len(data))), size=size, replace=False)
    res = [data.loc[i, "quote"] for i in _gen_id]
    data = data.drop(res, axis=0).reset_index(drop=True)
    return data, res


def generate_classification_example(class_quote_map: dict[str, list[str]]) -> str:
    res = []
    for cls, quotes in class_quote_map.items():
        for quote in quotes:
            res.append(f"Statement: {quote}\nCategory: {cls}")
    random.shuffle(res)
    return "\n".join(res)
