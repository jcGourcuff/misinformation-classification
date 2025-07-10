from os import makedirs
from os.path import dirname, isfile, join
from typing import Literal

from src.utils._serializer import ReferenceSerializer

pwd = dirname(__file__)

# general purpose directories
RAW_DATA_DIR = "./data/raw"
PROCCESSED_DIR = "./data/processed"


# Used to genereate accurate information
# pylint: disable=line-too-long
ReportAlias = Literal["physics_basis", "mitigation", "impact_risk_adaptation"]
IPCC_REPORTS: dict[ReportAlias, str] = {
    "physics_basis": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf",
    "mitigation": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_SPM.pdf",
    "impact_risk_adaptation": "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_SummaryForPolicymakers.pdf",
}
# Each paragraph extracted in a list
PROCESSED_IPCC_SECTIONS_FILE = "processed_ipcc_sections"
IPCC_DIR = join(RAW_DATA_DIR, "IPCC")
makedirs(IPCC_DIR, exist_ok=True)

# Misinformation dataset
QUOTA_CLIMAT_DATASET = "QuotaClimat/frugalaichallenge-text-train"
QUOTA_CLIMAT_DIR = join(RAW_DATA_DIR, "quota_climat")


# For classification
BINARY_CLS_DATASET_DIR = join(PROCCESSED_DIR, "./binary_cls_dataset")
makedirs(BINARY_CLS_DATASET_DIR, exist_ok=True)
BINARY_CLS_DATASET_FILE = join(BINARY_CLS_DATASET_DIR, "binary_cls_dataset.pkl.gz")

MULTI_CLS_DATASET_DIR = join(PROCCESSED_DIR, "./multi_cls_dataset")
makedirs(MULTI_CLS_DATASET_DIR, exist_ok=True)
MULTI_CLS_DATASET_FILE = join(MULTI_CLS_DATASET_DIR, "./multi_cls_dataset.pkl.gz")


# To hold utility files to interact with Mistral API
REQUEST_DIR = join(pwd, "../tmp/batch_requests")
makedirs(REQUEST_DIR, exist_ok=True)
JOB_ID_MAP_FILE = join(REQUEST_DIR, "file_name_job_id_map.yaml")
FILE_ID_MAP_FILE = join(REQUEST_DIR, "file_name_file_id_map.yaml")

if not isfile(JOB_ID_MAP_FILE):
    ReferenceSerializer.dump(data={}, file_path=JOB_ID_MAP_FILE)
if not isfile(FILE_ID_MAP_FILE):
    ReferenceSerializer.dump(data={}, file_path=FILE_ID_MAP_FILE)

# Fine tuning
FINETUNE_DATASET_FILE = join(
    MULTI_CLS_DATASET_FILE, "fine_tune_multi_cls_dataset.pkl.gz"
)
# for requests
FINETUNE_TRAIN_FILE = "fine_tune_multi_cls_dataset_train"
FINETUNE_VALIDATION_FILE = "fine_tune_multi_cls_dataset_validation"
