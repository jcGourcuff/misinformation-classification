import os

from mistralai import CompletionJobOut, Mistral, TrainingFileTypedDict

from src.conf import (
    FILE_ID_MAP_FILE,
    FINETUNE_TRAIN_FILE,
    FINETUNE_VALIDATION_FILE,
    JOB_ID_MAP_FILE,
)
from src.utils import ReferenceSerializer, logger


def launch_finetune_job(
    model: str,
    version: str,
    epochs: int,
) -> None:
    """
    Run batch inference with Mistral API.
    """

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    id_map = ReferenceSerializer.load(file_path=FILE_ID_MAP_FILE)

    logger.info(
        "Launching finetune job for training file %s, model %s, %s epochs",
        FINETUNE_TRAIN_FILE,
        model,
        epochs,
    )
    created_job = client.fine_tuning.jobs.create(
        model=model,
        training_files=[
            TrainingFileTypedDict(file_id=id_map[FINETUNE_TRAIN_FILE], weight=1)
        ],
        validation_files=[id_map[FINETUNE_VALIDATION_FILE]],
        suffix=f"tuned-{version}",
        hyperparameters={"learning_rate": 0.0001, "epochs": epochs},
        auto_start=True,
        job_type="completion",
    )

    if not isinstance(created_job, CompletionJobOut):
        raise ValueError("The job creation did not return a CompletionJobOut instance.")

    id_map = ReferenceSerializer.load(file_path=JOB_ID_MAP_FILE)
    id_map[f"{FINETUNE_TRAIN_FILE.strip('_train')}_{model}"] = created_job.id

    ReferenceSerializer.dump(data=id_map, file_path=JOB_ID_MAP_FILE)
