# Module redxclassifiers.evaluate

Module for evaluating and plotting the results of models

??? example "View Source"
        #!/usr/bin/env python3

        # -*- coding: utf-8 -*-

        """

        Module for evaluating and plotting the results of models

        """

        import logging

        import os

        from typing import Any, List, Optional, Tuple, Union

        import matplotlib.pyplot as plt

        import numpy as np

        import pandas as pd

        import plotly.express as px

        import scipy

        import seaborn as sns

        from matplotlib.pyplot import cm

        from sklearn.metrics import (

            mean_absolute_error,

            mean_absolute_percentage_error,

            mean_squared_error,

            r2_score,

            root_mean_squared_error,

        )

        log = logging.getLogger(__name__)



        def get_confusion_matrix(predictions: Optional[list] = None, known: Optional[list] = None, df: Optional[pd.DataFrame] = None, predicted_column_name="prediction", known_column_name="known", labels=(0,1)) -> dict:

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

                "TN": cm[0][0],

                "FP": cm[0][1],

                "FN": cm[1][0],

                "TP": cm[1][1],

            }

            return cm_dict

        def plot_confusion_matrix(cm: dict, labels: list = ["True Neg", "False Pos", "False Neg", "True Pos"], title: str = "Confusion Matrix") -> None:

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

            plt.show()

        def plot_roc_curve(y_true: list, y_scores: list, title: str = "ROC Curve") -> None:

            """

            Plot the ROC curve.

            Args:

                y_true (list): List of true values.

                y_scores (list): List of predicted scores.

                title (str): Title of the plot.

            Returns:

                None

            """

            from sklearn.metrics import roc_curve, auc

            fpr, tpr, _ = roc_curve(y_true, y_scores)

            roc_auc = auc(fpr, tpr)

            plt.figure()

            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")

            plt.xlabel("False Positive Rate")

            plt.ylabel("True Positive Rate")

            plt.title(title)

            plt.legend(loc="lower right")

            plt.savefig



        if __name__ == "__main__":

            import doctest

            doctest.testmod(verbose=True)

## Variables

```python3
log
```

## Functions


### get_confusion_matrix

```python3
def get_confusion_matrix(
    predictions: Optional[list] = None,
    known: Optional[list] = None,
    df: Optional[pandas.core.frame.DataFrame] = None,
    predicted_column_name='prediction',
    known_column_name='known',
    labels=(0, 1)
) -> dict
```

Get the confusion matrix for the predictions and known values.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| predictions | list | List of predicted values. | None |
| known | list | List of known values. | None |
| df | pd.DataFrame | DataFrame containing the predictions and known values. | None |
| predicted_column_name | str | Name of the column containing the predicted values. | None |
| known_column_name | str | Name of the column containing the known values. | None |
| labels | tuple | Tuple of labels for the confusion matrix. | None |

**Returns:**

| Type | Description |
|---|---|
| dict | Confusion matrix. |

??? example "View Source"
        def get_confusion_matrix(predictions: Optional[list] = None, known: Optional[list] = None, df: Optional[pd.DataFrame] = None, predicted_column_name="prediction", known_column_name="known", labels=(0,1)) -> dict:

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

                "TN": cm[0][0],

                "FP": cm[0][1],

                "FN": cm[1][0],

                "TP": cm[1][1],

            }

            return cm_dict


### plot_confusion_matrix

```python3
def plot_confusion_matrix(
    cm: dict,
    labels: list = ['True Neg', 'False Pos', 'False Neg', 'True Pos'],
    title: str = 'Confusion Matrix'
) -> None
```

Plot the confusion matrix.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| cm | dict | Confusion matrix. | None |
| labels | list | List of labels for the confusion matrix. | None |
| title | str | Title of the plot. | None |

**Returns:**

| Type | Description |
|---|---|
| None | None |

??? example "View Source"
        def plot_confusion_matrix(cm: dict, labels: list = ["True Neg", "False Pos", "False Neg", "True Pos"], title: str = "Confusion Matrix") -> None:

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

            plt.show()


### plot_roc_curve

```python3
def plot_roc_curve(
    y_true: list,
    y_scores: list,
    title: str = 'ROC Curve'
) -> None
```

Plot the ROC curve.

**Parameters:**

| Name | Type | Description | Default |
|---|---|---|---|
| y_true | list | List of true values. | None |
| y_scores | list | List of predicted scores. | None |
| title | str | Title of the plot. | None |

**Returns:**

| Type | Description |
|---|---|
| None | None |

??? example "View Source"
        def plot_roc_curve(y_true: list, y_scores: list, title: str = "ROC Curve") -> None:

            """

            Plot the ROC curve.

            Args:

                y_true (list): List of true values.

                y_scores (list): List of predicted scores.

                title (str): Title of the plot.

            Returns:

                None

            """

            from sklearn.metrics import roc_curve, auc

            fpr, tpr, _ = roc_curve(y_true, y_scores)

            roc_auc = auc(fpr, tpr)

            plt.figure()

            plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")

            plt.xlabel("False Positive Rate")

            plt.ylabel("True Positive Rate")

            plt.title(title)

            plt.legend(loc="lower right")

            plt.savefig
