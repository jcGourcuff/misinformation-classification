import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# pylint: disable=unsubscriptable-object,unsupported-assignment-operation
def build_metrics_from_confusion(confusion_mat: pd.DataFrame) -> pd.DataFrame:
    metrics = pd.DataFrame(index=confusion_mat.index)

    metrics["TP"] = np.diag(confusion_mat.values)
    metrics["FN"] = confusion_mat.sum(axis=1) - metrics["TP"]
    metrics["FP"] = confusion_mat.sum(axis=0) - metrics["TP"]
    metrics["TN"] = confusion_mat.sum().sum() - (
        metrics["TP"] + metrics["FN"] + metrics["FP"]
    )
    try:
        metrics = metrics.drop("not relevant")
    except KeyError:
        pass

    metrics["Precision"] = (
        100 * metrics["TP"] / (metrics["TP"] + metrics["FP"])
    ).round(2)
    metrics["Recall"] = (100 * metrics["TP"] / (metrics["TP"] + metrics["FN"])).round(2)
    metrics["F1-Score"] = (
        2
        * (metrics["Precision"] * metrics["Recall"])
        / (metrics["Precision"] + metrics["Recall"])
    ).round(2)

    metrics.loc["Average"] = metrics.mean(axis=0).round(2)

    return metrics[["Precision", "Recall", "F1-Score"]]


def get_confusion_matrix(prediction: pd.DataFrame) -> pd.DataFrame:
    labels = sorted(
        set(prediction["label"].unique()) | set(prediction["predicted_label"].unique())
    )
    res = pd.DataFrame(
        confusion_matrix(
            prediction["label"],
            prediction["predicted_label"],
            labels=labels,
        )
    )
    res.index = labels
    res.columns = labels
    return res
