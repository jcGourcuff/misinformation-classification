from os.path import dirname, join
from typing import Literal

PWD = dirname(__file__)

Prompt = Literal["multi_cls_zero_shot", "binary_cls", "multi_cls_few_shots"]


def load_prompt(
    prompt_name: Prompt,
) -> str:
    file = join(PWD, f"{prompt_name}.txt")

    with open(file, encoding="utf-8") as f:
        prompt = f.read()

    return prompt
