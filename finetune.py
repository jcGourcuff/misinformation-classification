"""
Mock file to illustrate how the finetune pipeline was ran.
(As I didn't think it made sens to include it in a recurrent workflow)
"""

from src.mistral.finetune import launch_finetune_job
from src.mistral.prep import generate_finetune_request_files
from src.processing.task import build_finetune_dataset


def finetune_multi_cls_task(
    model: str,
    version: str,
    epochs: int,
    mock: bool = True,
):
    if mock:  # to avoid unwanted jobs
        return

    # build dataset
    build_finetune_dataset()

    # prep files for uplaod
    generate_finetune_request_files()

    launch_finetune_job(
        model=model,  # choose model, eg mistral-3b-latest
        version=version,  # eg "v3", for iteration and new model naming
        epochs=epochs,
    )
