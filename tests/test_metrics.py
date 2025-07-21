import numpy as np
import pytest

from src.core.metrics import (
  accuracy_score,
  confusion_matrix,
  f1_score,
  precision_score,
  recall_score,
  roc_auc_score,
)


@pytest.mark.parametrize(
  "y_true,y_pred,expected",
  [
    (np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), 1.0),
    (np.array([0, 1, 0, 1]), np.array([1, 1, 0, 0]), 0.5),
  ],
)
def test_accuracy_score(y_true, y_pred, expected):
  assert np.isclose(accuracy_score(y_true, y_pred), expected)


@pytest.mark.parametrize(
  "y_true,y_pred,expected",
  [
    (np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), (1 + 2 / 3) / 2),  # P₀=1, P₁=⅔
    (np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]), ((2 / 3) + 1.0) / 2),  # P₀=⅔, P₁=1
  ],
)
def test_precision_score(y_true, y_pred, expected):
  assert np.isclose(precision_score(y_true, y_pred), expected)


@pytest.mark.parametrize(
  "y_true,y_pred,expected",
  [
    (np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1]), (0.5 + 1.0) / 2),  # R₀=½, R₁=1
    (np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0]), (1.0 + 0.5) / 2),  # R₀=1, R₁=½
  ],
)
def test_recall_score(y_true, y_pred, expected):
  assert np.isclose(recall_score(y_true, y_pred), expected)


@pytest.mark.parametrize(
  "y_true,y_pred",
  [
    (np.array([0, 1, 0, 1]), np.array([0, 1, 1, 1])),
    (np.array([1, 1, 0, 0]), np.array([1, 0, 0, 0])),
  ],
)
def test_f1_score_consistency(y_true, y_pred):
  P = precision_score(y_true, y_pred)
  R = recall_score(y_true, y_pred)
  expected = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
  assert np.isclose(f1_score(y_true, y_pred), expected)


def test_confusion_matrix():
  y_true = np.array([0, 1, 2, 1, 0])
  y_pred = np.array([0, 2, 1, 1, 0])
  expected = np.array([[2, 0, 0], [0, 1, 1], [0, 1, 0]])
  np.testing.assert_array_equal(confusion_matrix(y_true, y_pred), expected)


def test_roc_auc_binary():
  y_true = np.array([0, 1, 0, 1])
  y_score = np.array([0.2, 0.9, 0.1, 0.8])
  assert np.isclose(roc_auc_score(y_true, y_score), 1.0)


def test_roc_auc_multilabel():
  y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]])
  y_score = np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
  assert np.isclose(roc_auc_score(y_true, y_score), 1.0)
