import uuid
from typing import Literal

from mistralai import TextChunk, UserMessage

from src.conf import FINETUNE_DATASET_FILE, MULTI_CLS_DATASET_FILE
from src.mistral.inference.batch import BatchedPrompt, BatchRequest
from src.prompts import Prompt, load_prompt
from src.utils import ReferenceSerializer, logger

from ._utils import get_example_string


def generate_multi_cls_request_file(
    file_name: str,
    dataset: Literal["global_eval", "validation"],
    examples: int = 1,
):
    if dataset == "global_eval":
        dataset_ = ReferenceSerializer.load(MULTI_CLS_DATASET_FILE)
    else:
        dataset_ = ReferenceSerializer.load(FINETUNE_DATASET_FILE)["validation"]

    prompt_file: Prompt = (
        "multi_cls_zero_shot" if examples == 0 else "multi_cls_few_shots"
    )
    prompt = load_prompt(prompt_file)

    example_string = None
    if examples > 0:
        dataset_, example_string = get_example_string(
            dataset=dataset,
            n_per_class=examples,
        )
        prompt = prompt.replace("{examples}", example_string)

    batch_elems = []
    for quote, label, idx in zip(dataset_["quote"], dataset_["label"], dataset_.index):
        batch_elems.append(
            BatchedPrompt(
                custom_id=f"{idx}_{label}_{str(uuid.uuid4())}",
                max_tokens=10,
                messages=[
                    UserMessage(
                        content=[
                            TextChunk(
                                text=prompt.format(quote=quote),
                            ),
                        ],
                    ),
                ],
            )
        )

    logger.info("Generated %d prompts for %s", len(batch_elems), file_name)
    BatchRequest(prompts=batch_elems).to_jsonl(file_name=file_name)
