import uuid
from os.path import dirname, join

import pandas as pd

from src.inference import BatchedPrompt, BatchRequest, get_batch_job_result
from src.serializer import ReferenceSerializer

from ._build_data_set import BINARY_CLS_DATASET_DIR


def generate_batch_file_for_bin_cls(file_name: str):

    dataset = ReferenceSerializer.load(
        join(BINARY_CLS_DATASET_DIR, "binary_cls_dataset.pkl.gz")
    )

    prompt = open(join(dirname(__file__), "prompt.txt"), encoding="utf-8").read()
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
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt.format(
                                    examples=dataset["example_string"],
                                    quote=quote,
                                ),
                            },
                        ],
                    },
                ],
            )
        )

    print(f"Generated {len(batch_elems)} prompts for {file_name}")
    BatchRequest(prompts=batch_elems).to_jsonl(file_name=file_name)


def get_binary_cls_result(file_name: str, reload: bool = False):

    if reload:
        dataset = {
            "index": [],
            "true_label": [],
            "predicted_label": [],
        }
        for item in get_batch_job_result(file_name=file_name):
            split_custom_id = item["custom_id"].split("_")
            dataset["index"].append(split_custom_id[0])
            dataset["true_label"].append(split_custom_id[1])

            content = item["response"]["body"]["choices"][0]["message"]["content"]
            if content not in ["accurate statement", "misinformation"]:
                raise ValueError(
                    f"Unexpected content: {content}. Expected 'accurate statement' or 'misinformation'."
                )
            dataset["predicted_label"].append(content)
        dataset_as_df = pd.DataFrame(dataset).set_index("index")
        dataset_as_df.to_csv(
            join(BINARY_CLS_DATASET_DIR, f"{file_name}.csv"), index=True
        )
    return pd.read_csv(
        join(BINARY_CLS_DATASET_DIR, f"{file_name}.csv"), index_col="index"
    )
