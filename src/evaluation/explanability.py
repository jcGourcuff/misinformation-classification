"""
Supplementary info for binary classification tasks.
"""

from typing import Literal

import pandas as pd


def _get_breakdown(data: pd.DataFrame, context: Literal["context_1", "context_2"]):
    data = data.copy()
    data["count"] = 1

    per_context = data[[context, "count"]].groupby(context).sum().reset_index()

    per_context["% of predictions"] = (
        100 * per_context["count"] / sum(per_context["count"])
    ).round(2)

    return (
        per_context.rename(
            columns={
                context: "context",
                "count": "count of predictions",
                "% of predictions": "% of predictions",
            }
        )
        .sort_values(
            by="% of predictions",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def get_breakdown_per_contexts(preddictions: pd.DataFrame):
    preddictions = preddictions.copy()

    accurate_statements = preddictions[preddictions["label"] == "accurate statement"]

    accurate_per_personae = _get_breakdown(accurate_statements, "context_1").rename(
        columns={
            "context": "personae",
        }
    )
    accurate_per_emotion = _get_breakdown(accurate_statements, "context_2").rename(
        columns={
            "context": "emotion",
        }
    )

    misinformation_per_per_sub_class = _get_breakdown(
        preddictions[preddictions["label"] == "misinformation"], "context_1"
    ).rename(
        columns={
            "context": "misinformation sub-class",
        }
    )

    return accurate_per_personae, accurate_per_emotion, misinformation_per_per_sub_class
