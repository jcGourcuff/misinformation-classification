import re
from os.path import isfile, join
from typing import get_args

import requests

from src.conf import IPCC_DIR, IPCC_REPORTS, PROCESSED_IPCC_SECTIONS_FILE, ReportAlias
from src.mistral.ocr import parse_report_with_ocr
from src.utils import ReferenceSerializer, logger


def load_and_process_ipcc_reports(reload: bool = False) -> list[str]:
    """
    Returns a list of sections from the IPCC reports.
    """
    file_path = join(IPCC_DIR, f"{PROCESSED_IPCC_SECTIONS_FILE}.pkl.gz")
    if reload or not isfile(file_path):
        for report in IPCC_REPORTS:
            _load_ipcc_report(report=report)

        _compile_ocr_processed_reports()
    else:
        logger.info(
            "IPCC reports already processed. Loading from %s",
            file_path,
        )
    return ReferenceSerializer.load(
        file_path=join(IPCC_DIR, f"{PROCESSED_IPCC_SECTIONS_FILE}.pkl.gz")
    )


def _load_ipcc_report(report: ReportAlias):
    pdf_file_path = join(IPCC_DIR, f"{report}.pdf")
    with open(pdf_file_path, "wb") as file:
        logger.info("Downloading IPPC's %s ", IPCC_REPORTS[report])
        data = requests.get(IPCC_REPORTS[report], timeout=10)
        file.write(data.content)

    parse_report_with_ocr(report)


def _compile_ocr_processed_reports():
    all_entries = []
    for report in get_args(ReportAlias):
        all_entries.extend(_get_entries_from_ocr_processed(report=report))

    ReferenceSerializer.dump(
        data=all_entries,
        file_path=join(IPCC_DIR, f"{PROCESSED_IPCC_SECTIONS_FILE}.pkl.gz"),
    )


def _get_entries_from_ocr_processed(report: ReportAlias) -> list[str]:
    file = ReferenceSerializer.load(
        file_path=join(IPCC_DIR, f"ocr_processed_{report}.pkl.gz")
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
