import os
from typing import Literal

from mistralai import Any, Mistral, ResponseFormat


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
