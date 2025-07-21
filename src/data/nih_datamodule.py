import argparse
import json
import random
from collections import defaultdict

from src.data.datamodule import DataModule
from src.data.nih_cxr import DataLoader, NIHChestDataset


def _stratified_split(indices, labels, test_size, seed):
  """A manual implementation of a stratified split."""
  random.seed(seed)

  # Group indices by the primary class for stratification
  label_to_indices = defaultdict(list)
  for idx, label_vec in zip(indices, labels):
    primary_label = label_vec.argmax()
    label_to_indices[primary_label].append(idx)

  train_indices = []
  test_indices = []

  # For each class, split the indices proportionally
  for _, idx_list in label_to_indices.items():
    random.shuffle(idx_list)
    n_test = int(len(idx_list) * test_size)
    test_indices.extend(idx_list[:n_test])
    train_indices.extend(idx_list[n_test:])

  random.shuffle(train_indices)
  random.shuffle(test_indices)
  return train_indices, test_indices


class NIHDataModule(DataModule):
  def __init__(self, args: argparse.Namespace):
    super().__init__(args)
    self.image_dir = args.image_dir
    self.csv_file = args.csv_file
    self.batch_size = args.batch_size
    self.seed = args.seed

  def setup(self, stage: str | None = None):
    full_dataset = NIHChestDataset(
      image_directory=self.image_dir, csv_file=self.csv_file, augment=False
    )
    all_indices = list(range(len(full_dataset)))
    all_labels = [s[1] for s in full_dataset.samples]

    # Create the main train/test split
    train_val_indices, test_indices = _stratified_split(
      all_indices, all_labels, test_size=0.2, seed=self.seed
    )
    self.test_dataset = NIHChestDataset(
      image_directory=self.image_dir,
      csv_file=self.csv_file,
      indices=test_indices,
      augment=False,
    )

    # Create the train/validation split from the remaining data
    train_val_labels = [all_labels[i] for i in train_val_indices]
    train_indices, val_indices = _stratified_split(
      train_val_indices, train_val_labels, test_size=0.2, seed=self.seed
    )

    self.train_dataset = NIHChestDataset(
      image_directory=self.image_dir,
      csv_file=self.csv_file,
      indices=train_indices,
      augment=True,
    )
    self.val_dataset = NIHChestDataset(
      image_directory=self.image_dir,
      csv_file=self.csv_file,
      indices=val_indices,
      augment=False,
    )

  def train_dataloader(self) -> DataLoader:
    return DataLoader(
      self.train_dataset, batch_size=self.batch_size, shuffle=True, seed=self.seed
    )

  def val_dataloader(self) -> DataLoader:
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)

  def test_dataloader(self) -> DataLoader:
    return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

  @staticmethod
  def add_argparse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    with open("config.jsonc", "r") as config_file:
      config = json.load(config_file)["train"]
      parser.add_argument(
        "--image-dir",
        type=str,
        help="Path to the directory containing the images.",
        default=config.get("image_dir"),
        required=config.get("image_dir") is None,
      )
      parser.add_argument(
        "--csv-file",
        type=str,
        default=config.get("csv_file"),
        required=config.get("csv_file") is None,
        help="Path to the CSV metadata file.",
      )
    return parser
