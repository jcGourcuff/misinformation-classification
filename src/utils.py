import os
from typing import Literal

from mistralai import Any, Mistral, ResponseFormat


def _load_api_key(file_path: str = "./env/MISTRAL_API_KEY") -> None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            api_key = file.read().strip()
        os.environ["MISTRAL_API_KEY"] = api_key
    except FileNotFoundError as err:
        raise FileNotFoundError(f"API key file not found: {file_path}") from err


# pylint: disable=too-many-positional-arguments
def run_mistral(
    messages: list[dict[str, Any]],
    model: Literal["mistral-large-latest"],
    response_format: ResponseFormat | None = None,
    presence_penalty: float = 0.0,
    max_tokens: int = 2048,
    temperature: float | None = None,
    n_completion: int | None = None,
) -> Any:
    _load_api_key()
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    chat_response = client.chat.complete(
        model=model,
        messages=messages,
        response_format=response_format,
        max_tokens=max_tokens,
        random_seed=42,
        presence_penalty=presence_penalty,
        temperature=temperature,
        n=n_completion,
    )

    return chat_response
