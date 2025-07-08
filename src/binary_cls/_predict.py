def gen_batch_request(
    model: str,
    mode: str = "fim",
    job_type: str = "binary_cls",
    batch_size: int = 10,
) -> dict:
    """
    Generate a batch request for binary classification.
    """
    return {
        "model": model,
        "mode": mode,
        "job_type": job_type,
        "batch_size": batch_size,
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    }
