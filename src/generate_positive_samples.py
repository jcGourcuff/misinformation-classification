import json
import logging
import os
import uuid
from os.path import join
from typing import Literal

from mistralai import ResponseFormat

from src.load_raw_data import IPCC_REPORTS
from src.serializer import ReferenceSerializer
from src.utils import run_mistral

POSITIVE_SAMPLES_DIR = "./data/synthetic_positive_samples"


def get_true_quotes_prompt() -> str:
    return """
## Context

The IPCC reports provide a comprehensive overview of the current 
state of climate science, including the physical basis of climate 
change, its impacts, and mitigation strategies.

## Task

---
1. You are tasked with generating a total of {n_quotes} quotes relying on facts 
extracted from the IPCC report given in as a document without explicitly citing the IPCC.
You should provide the quote without the context, only what the person wrote/said.
 
---
2. These quotes should be in the form of quotes from a made up debate, 
an informal discussion, a news paper column, a scientific article, or any other form that
you can come up with. Here are some of how to embed the quotes in a quote:

Example 1:
I fear, however, that sinister forces and the dark arts are being mustered to make sure 
[the farmers’ protests] lose. Which is why, if you can actually find a report about the dispute, 
there’s no sympathy for poor old Farmer Giles. Because he’s being labelled as Hitler in a tractor veryone
from the BBC to The Guardian is saying that the farmer protests in Germany have nothing to do with fuel prices and are, 
in fact, a smokescreen for a resurgence of the far right. They imply that if you peer over the steaming piles of manure being 
left at the Brandenburg Gate, you can see lots of weird rural boys in brown shirts with daggers and suspiciously neat hair.

Example 2:
Global Warming? Tell that to the southern districts that woke up to negative 10 degrees this morning.

Example 3:
Interior Secretary Ken Salazar’s speech tonight at the Democratic National Convention was an impressive example of rhetorical charades, 
bestowing credit upon President Obama for the bright horizon of America’s energy future despite the administration’s relentless, 
take-no-prisoners assault on affordable domestic coal, oil, and natural gas.

---
3. The output should be a JSON object with a single key "quotes", which is an array of strings.
The output should be in the following format:

{{
    "quotes": [
        "Quote 1",
        "Quote 2",
        "Quote 3",
        "Quote 4",
        "Quote 5",
        "Quote 6",
        "Quote 7",
        "Quote 8",
        "Quote 9",
        "Quote 10"
    ]
}}

"""


def get_true_quotes_format() -> ResponseFormat:
    return ResponseFormat(
        json_schema={
            "type": "object",
            "properties": {
                "quotes": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "description": "A generated statement based on IPCC reports.",
                    },
                }
            },
            "required": ["quotes"],
        }
    )


def generate_true_quotes_with_ipcc_report(
    n_quotes: int,
    ipcc_report: Literal["physics_basis", "mitigation", "impact_risk_adaptation"],
    n_completion: int = 1,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": get_true_quotes_prompt().format(n_quotes=n_quotes),
                },
                {
                    "type": "document_url",
                    "document_url": IPCC_REPORTS[ipcc_report],
                },
            ],
        },
    ]
    res = run_mistral(
        messages=messages,
        model="mistral-large-latest",
        temperature=1.0,
        max_tokens=10000,
        presence_penalty=2.0,
        n_completion=n_completion,
        response_format=get_true_quotes_format(),
    )

    os.makedirs("./tmp", exist_ok=True)
    generation_id = f"{n_quotes*n_completion}-{ipcc_report}-{str(uuid.uuid4())}"

    ReferenceSerializer.dump(
        data=res,
        file_path=join("./tmp", f"{generation_id}.pkl.gz"),
    )

    # concatenate_and_save_as_yaml(generation_id)
    logging.info(
        "Generated %s quotes for %s report with ID: %s",
        n_quotes * n_completion,
        ipcc_report,
        generation_id,
    )


def concatenate_and_save_as_yaml(generation_id: str):
    os.makedirs(POSITIVE_SAMPLES_DIR, exist_ok=True)
    file_path = join("./tmp", f"{generation_id}.pkl.gz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")

    res = ReferenceSerializer.load(file_path=file_path)
    all_quotes = []
    for completion in res.choices:
        content = completion.message.content.strip("```").strip("json")
        try:
            as_json = json.loads(content)
        except json.JSONDecodeError as err:
            raise ValueError(
                f"{content}\nFailed to decode JSON from {generation_id}."
            ) from err
        all_quotes.extend(as_json["quotes"])
    ReferenceSerializer.dump(
        data=all_quotes,
        file_path=join(POSITIVE_SAMPLES_DIR, f"{generation_id}.yaml"),
    )
