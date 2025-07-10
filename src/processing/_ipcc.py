import json
import os
import re
from os import makedirs
from os.path import join
from pathlib import Path
from typing import Literal

import requests
from mistralai import DocumentURLChunk, Mistral

from src.serializer import ReferenceSerializer
from src.utils import load_api_key

# pylint: disable=line-too-long
ReportAlias = Literal["physics_basis", "mitigation", "impact_risk_adaptation"]
IPCC_REPORTS: dict[ReportAlias, str] = {
    "physics_basis": "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_SPM.pdf",
    "mitigation": "https://www.ipcc.ch/report/ar6/wg3/downloads/report/IPCC_AR6_WGIII_SPM.pdf",
    "impact_risk_adaptation": "https://www.ipcc.ch/report/ar6/wg2/downloads/report/IPCC_AR6_WGII_SummaryForPolicymakers.pdf",
}


def load_and_process_ipcc_reports(work_dir: str, reload: bool = False) -> list[str]:
    if reload:
        for report in IPCC_REPORTS:
            _load_ipcc_report(work_dir=work_dir, report=report)

        _compile_ocr_processed_reports(work_dir=work_dir)
    return ReferenceSerializer.load(
        file_path=join(work_dir, "processed_IPCC_sections.pkl.gz")
    )


def _load_ipcc_report(work_dir: str, report: ReportAlias):
    makedirs(work_dir, exist_ok=True)

    pdf_file_path = join(work_dir, f"{report}.pdf")
    with open(pdf_file_path, "wb") as file:
        data = requests.get(IPCC_REPORTS[report], timeout=10)
        file.write(data.content)

    _parse_report_with_ocr(report, work_dir=work_dir)


def _compile_ocr_processed_reports(work_dir: str):
    all_entries = []
    for report in ["physics_basis", "mitigation", "impact_risk_adaptation"]:
        all_entries.extend(
            _get_entries_from_ocr_processed(report=report, work_dir=work_dir)
        )

    ReferenceSerializer.dump(
        data=all_entries,
        file_path=join(work_dir, "processed_IPCC_sections.pkl.gz"),
    )


def _get_entries_from_ocr_processed(report: str, work_dir: str) -> list[str]:
    file = ReferenceSerializer.load(
        file_path=join(work_dir, f"ocr_processed_{report}.pkl.gz")
    )

    pattern = r"\n[A-Z]\.\d+\.\d+"
    result: list[str] = []
    for page in file["pages"]:
        if not re.search(pattern, page["markdown"]):
            continue
        for k, elem in enumerate(re.split(pattern, page["markdown"])):
            elem = elem.split("\n\n")[0]
            if len(result) > 0 and k == 0:
                result[-1] += elem
            else:
                result.append(elem)
    return result


def _parse_report_with_ocr(report: ReportAlias, work_dir: str) -> None:
    load_api_key()
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    file_name = join(work_dir, f"{report}.pdf")
    file = Path(file_name)
    assert file.is_file()

    uploaded_file = client.files.upload(
        file={
            "file_name": file.stem,
            "content": file.read_bytes(),
        },
        purpose="ocr",
    )

    signed_url = client.files.get_signed_url(file_id=uploaded_file.id, expiry=1)

    pdf_response = client.ocr.process(
        document=DocumentURLChunk(document_url=signed_url.url),
        model="mistral-ocr-latest",
        include_image_base64=False,
    )

    response_dict = json.loads(pdf_response.model_dump_json())

    ReferenceSerializer.dump(
        data=response_dict,
        file_path=join(work_dir, f"ocr_processed_{report}.pkl.gz"),
    )
