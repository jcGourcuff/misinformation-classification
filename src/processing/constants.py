from os import makedirs
from os.path import join
from typing import Literal

# general purpose dir
RAW_DATA_DIR = "./data/raw"


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
