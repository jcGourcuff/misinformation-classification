import os
from typing import Literal

from mistralai import Mistral

from src.classification.multiclass._build_data_set import LABEL_MAP
from src.inference._batch_inference import FILE_ID_MAP_FILE, JOB_ID_MAP_FILE
from src.serializer import ReferenceSerializer
from src.utils import load_api_key

load_api_key()


def launch_fine_tune_job(
    file_name_train: str,
    file_name_validation: str,
    model: str,
    version: str,
    epochs: int,
) -> None:
    """
    Run batch inference with Mistral API.
    """

    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])

    id_map = ReferenceSerializer.load(file_path=FILE_ID_MAP_FILE)

    created_job = client.fine_tuning.jobs.create(
        model=model,
        training_files=[{"file_id": id_map[file_name_train], "weight": 1}],
        validation_files=[id_map[file_name_validation]],
        suffix=f"tuned-{version}",
        hyperparameters={"learning_rate": 0.0001, "epochs": epochs},
        auto_start=True,
        job_type="completion",
    )

    id_map = ReferenceSerializer.load(file_path=JOB_ID_MAP_FILE)
    id_map[f"{file_name_train.strip('_train')}_{model}"] = created_job.id

    ReferenceSerializer.dump(data=id_map, file_path=JOB_ID_MAP_FILE)
