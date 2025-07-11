import json
import os
from os.path import join
from pathlib import Path

from mistralai import DocumentURLChunk, Mistral

from src.conf import IPCC_DIR, ReportAlias
from src.utils import ReferenceSerializer, logger


def parse_report_with_ocr(report_alias: ReportAlias) -> None:
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
    file_name = join(IPCC_DIR, f"{report_alias}.pdf")

    file = Path(file_name)
    if not file.is_file():
        raise FileNotFoundError(
            f"File {file_name} does not exist. Please download the report first."
        )

    logger.info("Processing %s with OCR...", file_name)
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
        file_path=join(IPCC_DIR, f"ocr_processed_{report_alias}.pkl.gz"),
    )
    logger.info(
        "Done. Saved at %s", join(IPCC_DIR, f"ocr_processed_{report_alias}.pkl.gz")
    )
