import json
from typing import Literal

from src.conf import (
    FINETUNE_DATASET_FILE,
    FINETUNE_TRAIN_FILE,
    FINETUNE_VALIDATION_FILE,
)
from src.mistral.inference.batch import BatchRequest
from src.processing.reformat_data import reformat_jsonl
from src.prompts import load_prompt
from src.utils import ReferenceSerializer, logger


def generate_finetune_request_files():
    _generate_finetune_file(which="train")
    _generate_finetune_file(which="validation")


def _generate_finetune_file(which: Literal["train", "validation"]):

    file_name = FINETUNE_TRAIN_FILE if which == "train" else FINETUNE_VALIDATION_FILE
    file_path = BatchRequest.file_path(file_name)

    dataset = ReferenceSerializer.load(FINETUNE_DATASET_FILE)[which]

    prompt = load_prompt("multi_cls_zero_shot")
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
    logger.info("Generated %d prompts for %s", len(batch_elems), file_name)
