"""
Light-weight classification metrics using NumPy.

∘ accuracy_score       –   (ŷ==y).mean()                                        ─ O(N)
∘ precision_score      –   Σᵢ TPᵢ/(TPᵢ+FPᵢ) ÷ C                                 ─ O(N)
∘ recall_score         –   Σᵢ TPᵢ/(TPᵢ+FNᵢ) ÷ C                                 ─ O(N)
∘ f1_score             –   Σᵢ 2·Pᵢ·Rᵢ/(Pᵢ+Rᵢ) ÷ C                               ─ O(N)
∘ roc_auc_score        –   ∫₀¹ TPR(FPR) dFPR per label, macro-averaged          ─ O(N log N)
∘ confusion_matrix     –   counts of (y_true,y_pred) over all classes           ─ O(N)

C = number of classes.
"""

import numpy as np


def _binarize_multiclass(arr: np.ndarray, class_index: int):
  return (arr == class_index).astype(np.int32)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
  return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray):
  return (y_true == y_pred).mean()


def precision_score(y_true: np.ndarray, y_pred: np.ndarray):
  y_true = _ensure_2d(y_true)
  y_pred = _ensure_2d(y_pred)
  if y_true.shape[1] == 1:
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    precisions = []
    for label in unique_labels:
      t_p = np.sum((y_pred == label) & (y_true == label))
      f_p = np.sum((y_pred == label) & (y_true != label))
      precisions.append(t_p / (t_p + f_p) if (t_p + f_p) > 0 else 0.0)
    return np.mean(precisions)
  precisions = []
  for col in range(y_true.shape[1]):
    t_p = np.sum((y_pred[:, col] == 1) & (y_true[:, col] == 1))
    f_p = np.sum((y_pred[:, col] == 1) & (y_true[:, col] == 0))
    precisions.append(t_p / (t_p + f_p) if (t_p + f_p) > 0 else 0.0)
  return np.mean(precisions)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray):
  y_true = _ensure_2d(y_true)
  y_pred = _ensure_2d(y_pred)
  if y_true.shape[1] == 1:
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    recalls = []
    for label in unique_labels:
      t_p = np.sum((y_pred == label) & (y_true == label))
      f_n = np.sum((y_pred != label) & (y_true == label))
      recalls.append(t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0.0)
    return np.mean(recalls)
  recalls = []
  for col in range(y_true.shape[1]):
    t_p = np.sum((y_pred[:, col] == 1) & (y_true[:, col] == 1))
    f_n = np.sum((y_pred[:, col] == 0) & (y_true[:, col] == 1))
    recalls.append(t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0.0)
  return np.mean(recalls)


def f1_score(y_true: np.ndarray, y_pred: np.ndarray):
  precision_macro = precision_score(y_true, y_pred)
  recall_macro = recall_score(y_true, y_pred)
  return (
    2 * precision_macro * recall_macro / (precision_macro + recall_macro)
    if (precision_macro + recall_macro) > 0
    else 0.0
  )


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray, average: str = "macro"):
  y_true = _ensure_2d(y_true)
  y_score = _ensure_2d(y_score)
  class_count = y_true.shape[1]
  auc_values: list[float] = []
  for col in range(class_count):
    true_col = y_true[:, col]
    score_col = y_score[:, col]
    thresholds = np.unique(score_col)[::-1]
    tpr_values = [0.0]
    fpr_values = [0.0]
    for θ in thresholds:
      positive_pred = score_col >= θ
      t_p = np.sum((true_col == 1) & positive_pred)
      f_p = np.sum((true_col == 0) & positive_pred)
      f_n = np.sum((true_col == 1) & ~positive_pred)
      t_n = np.sum((true_col == 0) & ~positive_pred)
      tpr_values.append(t_p / (t_p + f_n) if (t_p + f_n) > 0 else 0.0)
      fpr_values.append(f_p / (f_p + t_n) if (f_p + t_n) > 0 else 0.0)
    auc_values.append(np.trapezoid(tpr_values, fpr_values))
  if average == "macro":
    return np.mean(auc_values)
  else:
    raise ValueError(f"Unimplemented average method: {average}")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray):
  y_true = y_true.flatten()
  y_pred = y_pred.flatten()
  all_labels = np.unique(np.concatenate([y_true, y_pred]))
  label_to_index = {label: idx for idx, label in enumerate(all_labels)}
  matrix_dimension = len(all_labels)
  matrix = np.zeros((matrix_dimension, matrix_dimension), dtype=np.int64)
  for actual, predicted in zip(y_true, y_pred):
    i = label_to_index[actual]
    j = label_to_index[predicted]
    matrix[i, j] += 1
  return matrix
