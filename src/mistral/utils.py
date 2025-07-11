import os

from src.utils import logger


def load_api_key(file_path: str = "./env/MISTRAL_API_KEY") -> None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            api_key = file.read().strip()
        os.environ["MISTRAL_API_KEY"] = api_key
    except FileNotFoundError as err:
        logger.info(
            "Please provide an API key in a root/env/MISTRAL_API_KEY file.\n"
            "Where root is the root directory of the project."
        )
        raise FileNotFoundError(f"API key file not found at {file_path}") from err
