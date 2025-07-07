from os import makedirs
from os.path import join
from typing import Literal

import fitz
import pandas as pd
import requests
from datasets import load_dataset

QUOATA_CLIMAT_DATASET = "QuotaClimat/frugalaichallenge-text-train"

# pylint: disable=line-too-long
IPCC_REPORTS = {
    "physics_basis": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf",
    "mitigation": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_SPM.pdf",
    "impact_risk_adaptation": "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_SPM.pdf",
}


def load_quota_climat_dataset(work_dir: str, reload: bool = False) -> pd.DataFrame:
    makedirs(work_dir, exist_ok=True)

    if reload:
        dataset = load_dataset(QUOATA_CLIMAT_DATASET, cache_dir=work_dir)

        full_dataset = pd.concat(
            [dataset["train"].to_pandas(), dataset["test"].to_pandas()],
            ignore_index=True,
        )
        full_dataset.to_csv(join(work_dir, "quota_climat_dataset.csv"), index=False)

    try:
        return pd.read_csv(join(work_dir, "quota_climat_dataset.csv"))
    except FileNotFoundError as err:
        raise FileNotFoundError(
            "Dataset not found. Please run the script with reload=True to download it."
        ) from err


def _load_ipcc_report(
    work_dir: str,
    which: Literal["physics_basis", "mitigation", "impact_risk_adaptation"],
):
    makedirs(work_dir, exist_ok=True)
    pdf_file_path = join(work_dir, f"{which}.pdf")
    with open(pdf_file_path, "wb") as file:
        data = requests.get(IPCC_REPORTS[which], timeout=10)
        file.write(data.content)

    text = ""
    for page in fitz.open(pdf_file_path):
        text += page.get_text()

    with open(join(work_dir, f"{which}.txt"), "w", encoding="utf-8") as file:
        file.write(text)


def load_ipcc_reports_as_txt(work_dir: str, reload: bool = False) -> dict[str, str]:
    if reload:
        for which in IPCC_REPORTS:
            _load_ipcc_report(work_dir=work_dir, which=which)
    try:
        return {
            which: open(join(work_dir, f"{which}.txt"), "r", encoding="utf-8").read()
            for which in IPCC_REPORTS
        }
    except FileNotFoundError as err:
        raise FileNotFoundError(
            "IPCC reports not found. Please run the script with reload=True to download them."
        ) from err
