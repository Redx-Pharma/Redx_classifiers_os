#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module of unit tests for evaluation
"""

import logging

import pandas as pd
import pytest

from redxclassifiers.evaluate import get_confusion_matrix

log = logging.getLogger(__name__)


def test_get_confusion_matrix_with_lists():
    predictions = [0, 1, 1, 0, 1]
    known = [0, 1, 0, 0, 1]
    expected_output = {"TN": 2, "FP": 1, "FN": 0, "TP": 2}
    assert get_confusion_matrix(predictions=predictions, known=known) == expected_output


def test_get_confusion_matrix_with_dataframe():
    df = pd.DataFrame({"prediction": [0, 1, 1, 0, 1], "known": [0, 1, 0, 0, 1]})
    expected_output = {"TN": 2, "FP": 1, "FN": 0, "TP": 2}
    assert get_confusion_matrix(df=df) == expected_output


def test_get_confusion_matrix_invalid_lengths():
    predictions = [0, 1]
    known = [0, 1, 0]
    with pytest.raises(
        ValueError, match="Predictions and known values must have the same length."
    ):
        get_confusion_matrix(predictions=predictions, known=known)


def test_get_confusion_matrix_missing_inputs():
    with pytest.raises(
        ValueError, match="Either predictions or known values must be provided."
    ):
        get_confusion_matrix()


if __name__ == "__main__":
    pytest.main()
