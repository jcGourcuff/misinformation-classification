import uuid

from src.conf import BINARY_CLS_DATASET_FILE
from src.prompts import load_prompt
from src.utils import ReferenceSerializer, logger

from ..inference.batch import BatchedPrompt, BatchRequest
from ._utils import get_example_string


def generate_bin_cls_request_file(file_name: str):
    dataset = ReferenceSerializer.load(BINARY_CLS_DATASET_FILE)

    dataset, example_string = get_example_string(dataset=dataset, n_per_class=3)

    prompt = load_prompt("binary_cls")

    batch_elems = []
    for quote, label, idx in zip(
        dataset["quote"],
        dataset["label"],
        dataset.index,
    ):
        batch_elems.append(
            BatchedPrompt(
                custom_id=f"{idx}_{label}_{str(uuid.uuid4())}",
                max_tokens=10,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt.format(
                                    examples=example_string, quote=quote
                                ),
                            },
                        ],
                    },
                ],
            )
        )

    logger.info("Generated %d prompts for %s", len(batch_elems), file_name)
    BatchRequest(prompts=batch_elems).to_jsonl(file_name=file_name)
