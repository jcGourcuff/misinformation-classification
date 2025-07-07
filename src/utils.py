import os


def load_api_key(file_path: str = "./env/MISTRAL_API_KEY") -> None:
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            api_key = file.read().strip()
        os.environ["MISTRAL_API_KEY"] = api_key
    except FileNotFoundError as err:
        raise FileNotFoundError(f"API key file not found: {file_path}") from err
