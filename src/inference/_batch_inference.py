import json
import os
from dataclasses import dataclass
from os.path import join
from typing import Literal

from mistralai import AssistantMessage, Mistral, UserMessage

from src.serializer import ReferenceSerializer

WK_DIR = "./tmp/batch_requests"
MODELS = [
    "ministral-3b-latest",
    "ministral-8b-latest",
    "mistral-small-latest",
    "mistral-large-latest",
]
JOB_ID_MAP_FILE = join(WK_DIR, "file_name_job_id_map.yaml")
FILE_ID_MAP_FILE = join(WK_DIR, "file_name_file_id_map.yaml")

os.makedirs(WK_DIR, exist_ok=True)
if not os.path.isfile(JOB_ID_MAP_FILE):
    ReferenceSerializer.dump(data={}, file_path=JOB_ID_MAP_FILE)
if not os.path.isfile(FILE_ID_MAP_FILE):
    ReferenceSerializer.dump(data={}, file_path=FILE_ID_MAP_FILE)


@dataclass
class BatchedPrompt:
    custom_id: str
    max_tokens: int
    messages: list[AssistantMessage | UserMessage]
    temperature: float | None = None

    def format(self):
        json_res = {
            "custom_id": self.custom_id,
            "body": {
                "max_tokens": self.max_tokens,
                "random_seed": 42,
                "messages": self.messages,
            },
        }

        if self.temperature is not None:
            json_res["body"]["temperature"] = self.temperature  # type: ignore
        return json.dumps(json_res)


@dataclass
class BatchRequest:
    prompts: list[BatchedPrompt]

    @staticmethod
    def file_path(file_name: str) -> str:
        return join(WK_DIR, f"{file_name}.jsonl")

    def to_jsonl(self, file_name: str) -> None:
        """
        Save the batch request to a JSONL file.
        """

        os.makedirs(WK_DIR, exist_ok=True)
        file_path = BatchRequest.file_path(file_name)
        with open(file_path, "wb") as file:
            for prompt in self.prompts:
                file.write((f"{prompt.format()}\n".encode("utf-8")))

    @staticmethod
    def get_batch_request(file_name: str):
        """
        Load a batch request from a JSONL file.
        """
        return open(BatchRequest.file_path(file_name), "rb")


def upload_file(
    file_name: str, purpose: Literal["batch", "fine-tune"] = "batch"
) -> None:
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    batch_data = client.files.upload(
        file={
            "file_name": f"{file_name}.jsonl",
            "content": BatchRequest.get_batch_request(file_name),
        },
        purpose=purpose,
    )

    id_map = ReferenceSerializer.load(file_path=FILE_ID_MAP_FILE)
    id_map[file_name] = batch_data.id

    ReferenceSerializer.dump(data=id_map, file_path=FILE_ID_MAP_FILE)


def run_batch_mistral(
    file_name: str,
    model=str,
    mode: Literal["chat", "fim"] = "chat",
    job_type: str = "testing",
) -> None:
    """
    Run batch inference with Mistral API.
    """

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    id_map = ReferenceSerializer.load(file_path=FILE_ID_MAP_FILE)

    endpoint = "/v1/chat/completions" if mode == "chat" else "/v1/fim/completions"
    created_job = client.batch.jobs.create(
        input_files=[id_map[file_name]],
        model=model,
        endpoint=endpoint,
        metadata={"job_type": job_type},
    )

    id_map = ReferenceSerializer.load(file_path=JOB_ID_MAP_FILE)
    id_map[f"{file_name}_{model}"] = created_job.id

    ReferenceSerializer.dump(data=id_map, file_path=JOB_ID_MAP_FILE)


def get_batch_job_result(file_name: str, model: str | None = None) -> list[dict]:
    """
    Get the batch job by file name.
    """
    id_map = ReferenceSerializer.load(file_path=JOB_ID_MAP_FILE)

    file_name_ = file_name
    if model is not None:
        file_name_ += f"_{model}"

    job_id = id_map[file_name_]

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    retrieved_job = client.batch.jobs.get(job_id=job_id)

    if not isinstance(file_id := retrieved_job.output_file, str):
        raise ValueError(f"Invalid output file: {retrieved_job.output_file}")

    output_file_stream = client.files.download(file_id=file_id)

    res = []
    for line in output_file_stream.iter_lines():
        res.append(json.loads(line))

    return res
