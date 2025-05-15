#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module for evaluating and plotting the results of models
"""

import logging
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)


def get_confusion_matrix(
    predictions: Optional[list] = None,
    known: Optional[list] = None,
    df: Optional[pd.DataFrame] = None,
    predicted_column_name="prediction",
    known_column_name="known",
    labels=(0, 1),
) -> dict:
    """
    Get the confusion matrix for the predictions and known values.
    Args:
        predictions (list): List of predicted values.
        known (list): List of known values.
        df (pd.DataFrame): DataFrame containing the predictions and known values.
        predicted_column_name (str): Name of the column containing the predicted values.
        known_column_name (str): Name of the column containing the known values.
        labels (tuple): Tuple of labels for the confusion matrix.
    Returns:
        dict: Confusion matrix.
    """

    if df is not None:
        predictions = df[predicted_column_name].tolist()
        known = df[known_column_name].tolist()

    if predictions is None or known is None:
        raise ValueError("Either predictions or known values must be provided.")

    if len(predictions) != len(known):
        raise ValueError("Predictions and known values must have the same length.")

    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(known, predictions, labels=labels)
    cm_dict = {
        "TN": int(cm[0][0]),
        "FP": int(cm[0][1]),
        "FN": int(cm[1][0]),
        "TP": int(cm[1][1]),
    }
    return cm_dict


def plot_confusion_matrix(
    cm: dict,
    labels: list = ["True Neg", "False Pos", "False Neg", "True Pos"],
    title: str = "Confusion Matrix",
    filename: str = "confusion_mat.png",
) -> None:
    """
    Plot the confusion matrix.
    Args:
        cm (dict): Confusion matrix.
        labels (list): List of labels for the confusion matrix.
        title (str): Title of the plot.
    Returns:
        None
    """
    fig, ax = plt.subplots()
    sns.heatmap(
        np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]]),
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels[1:],
        yticklabels=labels[:2],
    )
    ax.set_title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(filename)


def plot_roc_curve(
    y_true: list,
    y_scores: list,
    title: str = "ROC Curve",
    filename: str = "roc_curve.png",
) -> None:
    """
    Plot the ROC curve.
    Args:
        y_true (list): List of true values.
        y_scores (list): List of predicted scores.
        title (str): Title of the plot.
    Returns:
        None
    """
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})"
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(filename)


def accuracy(y_true: list, y_pred: list) -> float:
    """
    Calculate the accuracy of the predictions.
    Args:
        y_true (list): List of true values.
        y_pred (list): List of predicted values.
    Returns:
        float: Accuracy of the predictions.
    """
    from sklearn.metrics import accuracy_score

    return accuracy_score(y_true, y_pred)


def precision(y_true: list, y_pred: list) -> float:
    """
    Calculate the precision of the predictions.
    Args:
        y_true (list): List of true values.
        y_pred (list): List of predicted values.
    Returns:
        float: Precision of the predictions.
    """
    from sklearn.metrics import precision_score

    return precision_score(y_true, y_pred)


def recall(y_true: list, y_pred: list) -> float:
    """
    Calculate the recall of the predictions.
    Args:
        y_true (list): List of true values.
        y_pred (list): List of predicted values.
    Returns:
        float: Recall of the predictions.
    """
    from sklearn.metrics import recall_score

    return recall_score(y_true, y_pred)


def f1_score(y_true: list, y_pred: list) -> float:
    """
    Calculate the F1 score of the predictions.
    Args:
        y_true (list): List of true values.
        y_pred (list): List of predicted values.
    Returns:
        float: F1 score of the predictions.
    """
    from sklearn.metrics import f1_score

    return f1_score(y_true, y_pred)


def plot_feature_importance(
    model: Any,
    feature_names: list,
    title: str = "Feature Importance",
    filename: str = "feature_importance.png",
) -> None:
    """
    Plot the feature importance of the model.
    Args:
        model (Any): Trained model.
        feature_names (list): List of feature names.
        title (str): Title of the plot.
    Returns:

        None
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure()
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(
        range(len(importances)), [feature_names[i] for i in indices], rotation=90
    )
    plt.ylabel("Importance")
    plt.xlabel("Features")
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == "__main__":
    import doctest

    doctest.testmod(verbose=True)
