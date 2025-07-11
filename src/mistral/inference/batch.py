import json
import os
from dataclasses import dataclass
from os.path import join
from typing import Literal

from mistralai import Mistral

from src.conf import FILE_ID_MAP_FILE, JOB_ID_MAP_FILE, REQUEST_DIR
from src.utils import ReferenceSerializer, logger


@dataclass
class BatchedPrompt:
    custom_id: str
    max_tokens: int
    messages: list[str]
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
        return join(REQUEST_DIR, f"{file_name}.jsonl")

    def to_jsonl(self, file_name: str) -> None:
        """
        Save the batch request to a JSONL file.
        """
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

    logger.info("Uploading file %s for purpose %s", file_name, purpose)
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
    job_type: str = "testing",
) -> None:
    """
    Run batch inference with Mistral API.
    """

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    id_map = ReferenceSerializer.load(file_path=FILE_ID_MAP_FILE)

    logger.info("Running batch job for file %s with model %s", file_name, model)
    created_job = client.batch.jobs.create(
        input_files=[id_map[file_name]],
        model=model,
        metadata={"job_type": job_type},
        endpoint="/v1/chat/completions",
    )

    id_map = ReferenceSerializer.load(file_path=JOB_ID_MAP_FILE)
    id_map[f"{file_name}_{model}"] = created_job.id

    ReferenceSerializer.dump(data=id_map, file_path=JOB_ID_MAP_FILE)


def get_batch_job_result(file_name: str) -> list[dict]:
    """
    Get the batch job by file name.
    """
    file_name = file_name.split("/")[-1].split(".")[0]
    id_map = ReferenceSerializer.load(file_path=JOB_ID_MAP_FILE)

    job_id = id_map[file_name]

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    logger.info("Retrieving job for file %s", file_name)
    retrieved_job = client.batch.jobs.get(job_id=job_id)

    if not isinstance(file_id := retrieved_job.output_file, str):
        raise ValueError(f"Invalid output file: {retrieved_job.output_file}")

    output_file_stream = client.files.download(file_id=file_id)

    res = []
    for line in output_file_stream.iter_lines():
        res.append(json.loads(line))

    return res
