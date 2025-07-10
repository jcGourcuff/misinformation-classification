import os
import uuid
from os.path import dirname, join

import pandas as pd

from src.inference import BatchedPrompt, BatchRequest, get_batch_job_result
from src.serializer import ReferenceSerializer

from ._build_data_set import LABEL_MAP, MULTI_CLS_DATASET_DIR

MULT_CLS_PROMPT_PATH = join(dirname(__file__), "prompt.txt")


# to refacto
def generate_batch_file_for_multi_cls(
    file_name: str,
    dataset: pd.DataFrame,
    prompt_path: str,
    example_string: str | None = None,
):

    prompt = open(prompt_path, encoding="utf-8").read()
    if example_string is not None:
        prompt = prompt.replace("{examples}", example_string)
    batch_elems = []
    for quote, label, idx in zip(dataset["quote"], dataset["label"], dataset.index):
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
                                "text": prompt.format(quote=quote),
                            },
                        ],
                    },
                ],
            )
        )

    print(f"Generated {len(batch_elems)} prompts for {file_name}")
    BatchRequest(prompts=batch_elems).to_jsonl(file_name=file_name)


def get_multi_cls_result(
    file_name: str, model: str, original_dataset: pd.DataFrame, reload: bool = False
):
    if reload or not os.path.isfile(join(MULTI_CLS_DATASET_DIR, f"{file_name}.csv")):
        dataset = {
            "index": [],
            "predicted_label": [],
        }
        for item in get_batch_job_result(file_name=file_name, model=model):
            content = item["response"]["body"]["choices"][0]["message"]["content"]
            authorized_labels = list(LABEL_MAP.values())
            if content not in authorized_labels:
                print(f"Invalid output: {content}.")
                continue

            dataset["predicted_label"].append(content)
            split_custom_id = item["custom_id"].split("_")
            dataset["index"].append(int(split_custom_id[0]))
        dataset_as_df = pd.DataFrame(dataset).set_index("index")

        # Join to old dataset
        dataset_as_df = dataset_as_df.join(original_dataset, how="left")

        dataset_as_df.to_csv(
            join(MULTI_CLS_DATASET_DIR, f"{file_name}.csv"), index=False
        )

    return pd.read_csv(join(MULTI_CLS_DATASET_DIR, f"{file_name}.csv"))


def get_example_string(dataset: pd.DataFrame, n__per_class: int):
    if dataset.empty:
        return None

    # Get a few examples from each class
    examples = []
    for label in dataset["label"].unique():
        class_examples = dataset[dataset["label"] == label].sample(
            n=n__per_class, replace=True
        )
        examples.extend(class_examples["quote"].tolist())

    return "\n".join(examples)
