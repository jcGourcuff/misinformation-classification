import json
import os
from dataclasses import dataclass
from os.path import join
from typing import Literal

from mistralai import BatchJobOut, Mistral

from src.serializer import ReferenceSerializer

WK_DIR = "./tmp/batch_requests"
MODELS = [
    "ministral-3b-latest",
    "ministral-8b-latest",
    "mistral-small-latest",
    "mistral-large-latest",
]
JOB_ID_MAP_FILE = join(WK_DIR, "file_name_job_id_map.yaml")

os.makedirs(WK_DIR, exist_ok=True)
if not os.path.isfile(JOB_ID_MAP_FILE):
    ReferenceSerializer.dump(data={}, file_path=JOB_ID_MAP_FILE)


@dataclass
class BatchedPrompt:
    custom_id: str
    max_tokens: int
    messages: list[dict[str, str]]
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
            json_res["body"]["temperature"] = self.temperature
        return json.dumps(json_res)


@dataclass
class BatchRequest:
    prompts: list[BatchedPrompt]

    def to_jsonl(self, file_name: str) -> None:
        """
        Save the batch request to a JSONL file.
        """

        os.makedirs(WK_DIR, exist_ok=True)
        file_path = join(WK_DIR, f"{file_name}.jsonl")
        with open(file_path, "wb") as file:
            for prompt in self.prompts:
                file.write((f"{prompt.format()}\n".encode("utf-8")))

    @staticmethod
    def get_batch_request(file_name: str):
        """
        Load a batch request from a JSONL file.
        """
        return open(join(WK_DIR, f"{file_name}.jsonl"), "rb")


def run_batch_mistral(
    file_name: str,
    model=str,
    mode: Literal["chat", "fim"] = "chat",
    job_type: str = "testing",
) -> BatchJobOut:
    """
    Run batch inference with Mistral API.
    """

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    batch_data = client.files.upload(
        file={
            "file_name": f"{file_name}.jsonl",
            "content": BatchRequest.get_batch_request(file_name),
        },
        purpose="batch",
    )

    endpoint = "/v1/chat/completions" if mode == "chat" else "/v1/fim/completions"
    created_job = client.batch.jobs.create(
        input_files=[batch_data.id],
        model=model,
        endpoint=endpoint,
        metadata={"job_type": job_type},
    )

    id_map = ReferenceSerializer.load(file_path=JOB_ID_MAP_FILE)
    id_map[file_name] = created_job.id

    ReferenceSerializer.dump(data=id_map, file_path=JOB_ID_MAP_FILE)


def get_batch_job_result(file_name: str) -> list[dict]:
    """
    Get the batch job by file name.
    """
    id_map = ReferenceSerializer.load(file_path=JOB_ID_MAP_FILE)
    job_id = id_map[file_name]

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    retrieved_job = client.batch.jobs.get(job_id=job_id)

    output_file_stream = client.files.download(file_id=retrieved_job.output_file)

    res = []
    for line in output_file_stream.iter_lines():
        res.append(json.loads(line))

    return res
