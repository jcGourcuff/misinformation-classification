import uuid
from os.path import dirname, join

import pandas as pd

from src.inference import BatchedPrompt, BatchRequest, get_batch_job_result
from src.serializer import ReferenceSerializer

from ._build_data_set import LABEL_MAP, MULTI_CLS_DATASET_DIR


# to refacto
def generate_batch_file_for_multi_cls(file_name: str):

    dataset = ReferenceSerializer.load(
        join(MULTI_CLS_DATASET_DIR, "multi_cls_dataset.pkl.gz")
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


def get_multi_cls_result(model: str, reload: bool = False):
    file_name = f"multi_cls_{model}"
    if reload:
        dataset = {
            "index": [],
            "predicted_label": [],
        }
        for item in get_batch_job_result(file_name=file_name):
            split_custom_id = item["custom_id"].split("_")
            dataset["index"].append(int(split_custom_id[0]))

            content = item["response"]["body"]["choices"][0]["message"]["content"]
            authorized_labels = list(LABEL_MAP.values()) + ["accurate statement"]
            if content not in authorized_labels:
                raise ValueError(
                    f"Unexpected content: {content}. Expected one of {', '.join(authorized_labels)}."
                )
            dataset["predicted_label"].append(content)
        dataset_as_df = pd.DataFrame(dataset).set_index("index")

        # Join to old dataset
        original_dataset = ReferenceSerializer.load(
            join(MULTI_CLS_DATASET_DIR, "multi_cls_dataset.pkl.gz")
        )["dataset"]
        dataset_as_df = dataset_as_df.join(original_dataset, how="left")

        dataset_as_df.to_csv(
            join(MULTI_CLS_DATASET_DIR, f"{file_name}.csv"), index=False
        )

    return pd.read_csv(join(MULTI_CLS_DATASET_DIR, f"{file_name}.csv"))
