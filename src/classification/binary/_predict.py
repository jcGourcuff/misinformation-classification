import uuid
from os.path import dirname, join

import pandas as pd
from mistralai import TextChunk, UserMessage

from src.inference import BatchedPrompt, BatchRequest, get_batch_job_result
from src.serializer import ReferenceSerializer

from ._build_data_set import BINARY_CLS_DATASET_DIR


def generate_batch_file_for_bin_cls(file_name: str):

    dataset = ReferenceSerializer.load(
        join(BINARY_CLS_DATASET_DIR, "binary_cls_dataset.pkl.gz")
    )
    with open(join(dirname(__file__), "prompt.txt"), encoding="utf-8") as f:
        prompt = f.read()
    batch_elems = []
    for quote, label, idx in zip(
        dataset["dataset"]["quote"],
        dataset["dataset"]["label"],
        dataset["dataset"].index,
    ):
        batch_elems.append(
            BatchedPrompt(
                custom_id=f"{idx}_{label}_{str(uuid.uuid4())}",
                max_tokens=10,
                messages=[
                    UserMessage(
                        content=[
                            TextChunk(
                                text=prompt.format(
                                    examples=dataset["example_string"],
                                    quote=quote,
                                ),
                            ),
                        ],
                    ),
                ],
            )
        )

    print(f"Generated {len(batch_elems)} prompts for {file_name}")
    BatchRequest(prompts=batch_elems).to_jsonl(file_name=file_name)


def get_binary_cls_result(model: str, reload: bool = False):
    file_name = f"binary_cls_{model}"
    if reload:
        dataset: dict = {
            "index": [],
            "predicted_label": [],
        }
        for item in get_batch_job_result(file_name=file_name, model=model):
            split_custom_id = item["custom_id"].split("_")
            dataset["index"].append(int(split_custom_id[0]))

            content = item["response"]["body"]["choices"][0]["message"]["content"]
            if content not in ["accurate statement", "misinformation"]:
                raise ValueError(
                    f"Unexpected content: {content}. "
                    "Expected 'accurate statement' or 'misinformation'."
                )
            dataset["predicted_label"].append(content)
        dataset_as_df = pd.DataFrame(dataset).set_index("index")

        # Join to old dataset
        original_dataset = ReferenceSerializer.load(
            join(BINARY_CLS_DATASET_DIR, "binary_cls_dataset.pkl.gz")
        )["dataset"]
        dataset_as_df = dataset_as_df.join(original_dataset, how="left")

        dataset_as_df.to_csv(
            join(BINARY_CLS_DATASET_DIR, f"{file_name}.csv"), index=False
        )

    return pd.read_csv(join(BINARY_CLS_DATASET_DIR, f"{file_name}.csv"))
