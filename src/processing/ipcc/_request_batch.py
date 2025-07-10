import os
import random
import uuid

from mistralai import (
    AssistantMessage,
    JSONSchema,
    ResponseFormat,
    TextChunk,
    UserMessage,
)

from src.mistral.inference.batch import BatchedPrompt, BatchRequest
from src.mistral.inference.simple import run_mistral
from src.utils.logging import logger

PWD = os.path.dirname(__file__)
EMOTIONS = ["angry", "serious", "delusional", "neutral", "sarcastic", "evasive"]
PERSONAE = ["climate_enthusiast", "scientist", "newbie"]


def generate_request_file_for_accurate_sample_gen(
    ipcc_report_blocks: list[str], file_name: str, n_prompt_per_block=3
):
    random.seed(42)
    batch_elems = []
    for block in ipcc_report_blocks:
        if len(block) < 100:
            continue
        for _ in range(n_prompt_per_block):
            personae = random.choice(PERSONAE)
            emotion = random.choice(EMOTIONS)
            prompt = _get_prompt(block, personae, emotion)
            batch_elems.append(
                BatchedPrompt(
                    custom_id=f"{personae}_{emotion}_{str(uuid.uuid4())}",
                    max_tokens=500,
                    temperature=0.7,
                    messages=[
                        UserMessage(
                            content=[
                                TextChunk(text=prompt),
                            ],
                        ),
                    ],
                )
            )
    logger.info("Generated %s prompts for %s", len(batch_elems), file_name)
    BatchRequest(prompts=batch_elems).to_jsonl(file_name=file_name)


def generate_accurate_data_sample(prompt: str) -> str:
    """
    For testing.
    """
    messages: list[UserMessage | AssistantMessage] = [
        UserMessage(
            content=[
                TextChunk(text=prompt),
            ],
        ),
    ]
    res = run_mistral(
        messages=messages,
        model="mistral-large-latest",
        temperature=0.7,
        max_tokens=500,
        response_format=_get_response_format(),
    )

    return res.choices[0].message.content.strip("```").strip("json")


def _load_personae(personae: str) -> str:
    with open(
        os.path.join(PWD, f"./personae/{personae}.txt"), encoding="utf-8"
    ) as file:
        personae_description = file.read().strip()
    return personae_description


def _get_prompt(text_block: str, personae: str, emotion: str) -> str:
    with open(os.path.join(PWD, "./prompt.txt"), encoding="utf-8") as f:
        prompt = f.read().format(
            personae=_load_personae(personae),
            emotion=emotion,
            text_block=text_block,
        )
    return prompt


def _get_response_format() -> ResponseFormat:
    return ResponseFormat(
        json_schema=JSONSchema(
            name="response",
            schema_definition={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "text",
                    }
                },
                "required": ["text"],
            },
        )
    )
