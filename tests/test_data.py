"""
Tests for:
∘ `DataLoader` – Batching & Shuffling
∘ `_stratified_split` – Class‑Proportional Train/Test Partition
"""

from collections import Counter

import numpy as np
import pytest

from src.data.nih_cxr import DataLoader
from src.data.nih_datamodule import _stratified_split


class MockDataset:
  """
  Tiny 3‑class toy set:
  ∘ Class 0: 60%
  ∘ Class 1: 30%
  ∘ Class 2: 10%
  Each sample is a (1 × 224 × 224) float32 image + one‑hot label (3,)
  """

  def __init__(self, length: int = 100):
    self.length = length
    self.samples = []

    for i in range(length):
      if i < 0.60 * length:
        y = np.array([1, 0, 0], dtype=np.float32)
      elif i < 0.90 * length:
        y = np.array([0, 1, 0], dtype=np.float32)
      else:
        y = np.array([0, 0, 1], dtype=np.float32)
      x = np.random.rand(1, 224, 224).astype(np.float32)
      self.samples.append((x, y))

  def __len__(self):
    return self.length

  def __getitem__(self, idx: int):
    return self.samples[idx]


@pytest.mark.parametrize(
  "batch_size,total,expected_batches",
  [(32, 100, 4), (20, 60, 3)],
)
def test_dataloader_batching(batch_size, total, expected_batches):
  ds = MockDataset(total)
  loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

  batches = list(loader)
  assert len(batches) == expected_batches

  # shapes
  assert batches[0][0].shape == (batch_size, 1, 224, 224)
  assert batches[0][1].shape == (batch_size, 3)

  tail = total - batch_size * (expected_batches - 1)
  assert batches[-1][0].shape == (tail, 1, 224, 224)
  assert batches[-1][1].shape == (tail, 3)


def test_dataloader_shuffle():
  """
  Two successive epochs → different order when `shuffle=True`.
  """
  ds = MockDataset(100)
  loader = DataLoader(ds, batch_size=10, shuffle=True, seed=42)

  first_batch_epoch1 = next(iter(loader))[0]
  first_batch_epoch2 = next(iter(loader))[0]

  assert not np.array_equal(first_batch_epoch1, first_batch_epoch2)


def test_stratified_split():
  """
  Verify that each class keeps ~20 % of its members in the test fold.
  """
  labels = []
  for i in range(100):
    if i < 70:
      labels.append(np.array([1, 0, 0], dtype=np.float32))
    elif i < 90:
      labels.append(np.array([0, 1, 0], dtype=np.float32))
    else:
      labels.append(np.array([0, 0, 1], dtype=np.float32))

  idx = list(range(100))
  train_idx, test_idx = _stratified_split(idx, labels, test_size=0.2, seed=42)

  assert len(train_idx) == 80
  assert len(test_idx) == 20

  test_counts = Counter(int(labels[i].argmax()) for i in test_idx)
  assert test_counts[0] == 14  # 70 × 0.2
  assert test_counts[1] == 4  # 20 × 0.2
  assert test_counts[2] == 2  # 10 × 0.2
