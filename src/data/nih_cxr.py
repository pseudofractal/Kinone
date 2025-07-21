import random
from csv import DictReader
from pathlib import Path
from typing import Sequence

import albumentations as A
import cv2
import numpy as np

DISEASES = (
  "Atelectasis",
  "Cardiomegaly",
  "Effusion",
  "Infiltration",
  "Mass",
  "Nodule",
  "Pneumonia",
  "Pneumothorax",
  "Consolidation",
  "Edema",
  "Emphysema",
  "Fibrosis",
  "Pleural_Thickening",
  "Hernia",
  "No Finding",
)

augmentations = A.Compose(
    [
      A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
      A.HorizontalFlip(p=0.5),
      A.Rotate(limit=10),
      A.CLAHE(clip_limit=2, p=0.5),
      A.RandomBrightnessContrast(p=0.5),
    ]
  )


class NIHChestDataset:
  
  def __init__(
    self,
    image_directory: str | Path,
    csv_file: str | Path,
    indices: Sequence[int] | None = None,
    augment: bool = True,
  ):
    self.image_directory = Path(image_directory)
    self.csv_file = Path(csv_file)
    self.augment = augment

    with self.csv_file.open() as metadata_file:
      reader = DictReader(metadata_file)
      rows: list[tuple[str, np.ndarray]] = []
      for row in reader:
        file_name = row["Image Index"]
        findings = row["Finding Labels"].split("|")
        label_vector = np.zeros(len(DISEASES), dtype=bool)
        for diagnosis in findings:
          diagnosis = diagnosis.strip()
          if diagnosis == "":
            continue
          try:
            label_vector[DISEASES.index(diagnosis)] = True
          except ValueError:
            pass
        rows.append((file_name, label_vector))

    self.samples = rows if indices is None else [rows[idx] for idx in indices]

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    file_name, label = self.samples[idx]
    image_path = self.image_directory / file_name
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
      raise FileNotFoundError(image_path)

    if self.augment:
      image = augmentations(image=image)["image"]
    else:
      image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

    image = image.astype(np.float32) / 255.0
    image = image[np.newaxis, ...]
    return image, label.copy()


class DataLoader:
  def __init__(
    self,
    dataset: NIHChestDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 42,
  ):
    self.dataset = dataset
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.seed = seed
    self.__order = list(range(len(self.dataset)))
    self.__epoch = 0

  def __iter__(self):
    if self.shuffle:
      random.seed(self.__epoch + self.seed)
      random.shuffle(self.__order)
      self.__epoch += 1
    for start in range(0, len(self.__order), self.batch_size):
      batch_indices = self.__order[start : start + self.batch_size]
      images, labels = zip(*(self.dataset[idx] for idx in batch_indices))
      batch_x = np.stack(images, axis=0)
      batch_y = np.stack(labels, axis=0)
      yield batch_x, batch_y
